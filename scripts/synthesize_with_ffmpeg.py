#!/usr/bin/env python3
"""
Synthesize existing scripts to M4B using ffmpeg directly (no pydub).
Reads scripts from data/output/<book>/scripts/, calls Kokoro TTS per segment,
uses ffmpeg to convert/combine, creates M4B with chapter markers.

Usage:
    uv run python3 scripts/synthesize_with_ffmpeg.py <source_epub>
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audify.readers.ebook import EpubReader
from audify.utils.text import get_file_name_title, break_text_into_sentences
from audify.utils.m4b_builder import write_metadata_header, append_chapter_metadata
from audify.utils.audio import AudioProcessor

BOOKS_DIR = Path("/home/rd24/Downloads/books_deepseek")
OUTPUT_BASE = Path("data/output")
KOKORO_URL = "http://localhost:8887/v1/audio/speech"
VOICE = os.environ.get("VOICE", "af_bella")


def synthesize_book(source_path: Path):
    """Full synthesize-only pipeline using ffmpeg."""
    book_name = get_file_name_title(source_path.stem)
    out_dir = OUTPUT_BASE / book_name
    scripts_dir = out_dir / "scripts"
    wav_dir = out_dir / "wav_files"
    mp3_dir = out_dir / "episodes"

    if not scripts_dir.exists():
        print(f"No scripts dir for {book_name}")
        return False

    # Read chapter titles
    titles_path = scripts_dir / "chapter_titles.json"
    chapter_titles = []
    if titles_path.exists():
        chapter_titles = json.loads(titles_path.read_text())

    wav_dir.mkdir(parents=True, exist_ok=True)
    mp3_dir.mkdir(parents=True, exist_ok=True)

    # Get sorted script files
    script_files = sorted(scripts_dir.glob("episode_*_script.txt"))
    print(f"Found {len(script_files)} scripts")

    # Phase 1: Synthesize each chapter to MP3
    for sf in script_files:
        ep_num = int(sf.stem.split("_")[1])
        script = sf.read_text()
        if not script.strip() or script.startswith("[ERROR"):
            print(f"  [{ep_num}] Skipping (error/empty)")
            continue

        mp3_path = mp3_dir / f"episode_{ep_num:03d}.mp3"
        if mp3_path.exists() and mp3_path.stat().st_size > 1000:
            print(f"  [{ep_num}] Already exists, skipping")
            continue

        title = chapter_titles[ep_num - 1] if chapter_titles and ep_num <= len(chapter_titles) else f"Chapter {ep_num}"
        print(f"  [{ep_num}] {title} ({len(script)} chars)...", end=" ", flush=True)

        # Split script into segments
        sentences = break_text_into_sentences(script)
        segments = []
        current = ""
        for s in sentences:
            if current and len(current + " " + s) > 200:
                segments.append(current.strip())
                current = s
            else:
                current = (current + " " + s) if current else s
        if current.strip():
            segments.append(current.strip())

        # Synthesize each segment
        seg_wavs = []
        for si, seg in enumerate(segments):
            seg_path = wav_dir / f"ep_{ep_num:03d}_seg{si:04d}.wav"
            if seg_path.exists():
                seg_wavs.append(seg_path)
                continue
            try:
                resp = requests.post(KOKORO_URL, json={
                    "model": "kokoro", "input": seg,
                    "voice": VOICE, "response_format": "wav",
                }, timeout=120)
                resp.raise_for_status()
                seg_path.write_bytes(resp.content)
                seg_wavs.append(seg_path)
            except Exception as e:
                print(f"TTS err: {e}")
                continue

        if not seg_wavs:
            print("FAIL (no audio)")
            continue

        # Combine segments into one MP3 using ffmpeg concat
        concat_file = wav_dir / f"concat_{ep_num:03d}.txt"
        concat_file.write_text("\n".join(f"file '{p.resolve()}'" for p in seg_wavs))

        try:
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-acodec", "mp3", "-b:a", "128k",
                str(mp3_path),
            ], capture_output=True, timeout=300)
            size = mp3_path.stat().st_size
            print(f"OK ({size//1024}KB)")
        except Exception as e:
            print(f"FAIL: {e}")
        finally:
            concat_file.unlink(missing_ok=True)

    # Phase 2: Create M4B with chapters
    print("\nCreating M4B...")
    mp3_files = sorted(mp3_dir.glob("episode_*.mp3"))
    if not mp3_files:
        print("No MP3 files!")
        return False

    # Create M4B using ffmpeg directly (concat+metadata in one step)
    final_m4b = out_dir / f"{book_name}.m4b"
    concat_list = out_dir / "concat_list.txt"
    
    # Write ffmpeg concat list
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in mp3_files)
    )
    
    # Write metadata file
    meta_path = out_dir / "chapters.txt"
    write_metadata_header(meta_path)
    start_ms = 0
    for mp3 in mp3_files:
        ep_num = int(mp3.stem.split("_")[1])
        title = chapter_titles[ep_num - 1] if chapter_titles and ep_num <= len(chapter_titles) else f"Chapter {ep_num}"
        dur = AudioProcessor.get_duration(str(mp3))
        start_ms = append_chapter_metadata(meta_path, title, start_ms, dur)
    
    # Build cover args
    cover = out_dir / "cover.jpg"
    cover_args = []
    if cover.exists():
        cover_args = ["-i", str(cover), "-map", "0:a", "-map", "2:v",
                     "-disposition:v", "attached_pic", "-c:v", "copy"]
    else:
        cover_args = ["-map", "0:a"]
    
    # Run ffmpeg: concat files + add metadata + cover in one pass
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", str(concat_list),
           "-i", str(meta_path),
           *cover_args,
           "-map_metadata", "1",
           "-c:a", "aac", "-b:a", "64k",
           "-f", "mp4",
           str(final_m4b)]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    concat_list.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    
    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr[-200:]}")
        return False
    
    file_size = final_m4b.stat().st_size // (1024*1024)
    print(f"\n✅ M4B created: {final_m4b.name} ({file_size}MB)")
    return True

    print(f"\n✅ M4B created: {final_m4b} ({final_m4b.stat().st_size//1024//1024}MB)")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python3 scripts/synthesize_with_ffmpeg.py <epub_path>")
        sys.exit(1)

    epub_path = Path(sys.argv[1])
    if not epub_path.exists():
        epub_path = BOOKS_DIR / "en" / sys.argv[1]
    
    success = synthesize_book(epub_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
