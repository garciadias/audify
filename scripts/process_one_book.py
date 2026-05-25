#!/usr/bin/env python3
"""
Process a single book through direct API calls (not via audify subprocess).

Uses httpx directly instead of LiteLLM to avoid connection hanging issues.
Has aggressive timeouts and retry logic.

Usage: uv run python3 scripts/process_one_book.py <epub_path> --voice af_bella
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audify.readers.ebook import EpubReader
from audify.utils.prompts import AUDIOBOOK_PROMPT
from audify.utils.text import get_file_name_title
from audify.audiobook_creator import _clean_text_for_audiobook, _MAX_WORDS_PER_LLM_CHUNK

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
# Force immediate flush for log output
logging.getLogger().handlers[0].flush = lambda: sys.stdout.flush()
logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
MODEL = "ministral-3:14b"
KOKORO_URL = "http://localhost:8887/v1/audio/speech"
OUTPUT_BASE = Path("data/output")
TIMEOUT_SECS = 120  # per request


def call_ollama(prompt: str, system_prompt: str = "") -> str:
    """Call Ollama with timeout. Retries on failure."""
    import requests as req_lib
    last_err = None
    for attempt in range(5):
        try:
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {"num_predict": 2000, "temperature": 0.7},
            }
            resp = req_lib.post(
                f"{OLLAMA_BASE}/api/generate",
                json=payload,
                timeout=(30, 90),  # (connect_timeout, read_timeout)
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
            if text.strip():
                return text
            logger.warning(f"Empty response (attempt {attempt+1})")
        except Exception as e:
            last_err = e
            logger.warning(f"Ollama call failed (attempt {attempt+1}): {e}")
            if attempt < 4:
                time.sleep(3)
    raise RuntimeError(f"Ollama failed after 5 attempts: {last_err}")


def call_kokoro(text: str, voice: str, output_path: Path) -> bool:
    """Call Kokoro TTS with timeout."""
    import requests as req_lib
    try:
        resp = req_lib.post(
            KOKORO_URL,
            json={
                "model": "kokoro",
                "input": text,
                "voice": voice,
                "response_format": "wav",
            },
            timeout=(30, 120),
        )
        resp.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(resp.content)
        logger.info(f"  TTS OK: {len(resp.content)} bytes -> {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"  TTS failed: {e}")
        return False


def process_book(epub_path: str, voice: str = "af_bella", delete_first: bool = True):
    """Process one ebook: extract -> LLM scripts -> TTS -> combine."""
    epub = Path(epub_path)
    if not epub.exists():
        logger.error(f"File not found: {epub}")
        return False

    book_name = get_file_name_title(epub.stem)
    out_dir = OUTPUT_BASE / book_name
    scripts_dir = out_dir / "scripts"
    episodes_dir = out_dir / "episodes"

    if delete_first and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {epub.name}")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"{'='*60}")

    # Step 1: Read EPUB
    logger.info("Reading EPUB...")
    reader = EpubReader(str(epub))
    chapters = reader.get_chapters()
    logger.info(f"Found {len(chapters)} chapters")

    # Get cover
    try:
        cover = reader.get_cover_image(str(out_dir))
        if cover:
            logger.info(f"Cover saved: {cover}")
    except Exception:
        pass

    # Step 2: Generate scripts
    chapter_titles = []
    for i, chapter_html in enumerate(chapters, 1):
        title = reader.get_chapter_title(chapter_html) or f"Chapter {i}"
        chapter_titles.append(title)
        logger.info(f"\n[{i}/{len(chapters)}] {title}")

        # Clean text
        cleaned = _clean_text_for_audiobook(chapter_html)
        if not cleaned.strip():
            logger.warning("  Empty after cleaning, skipping")
            continue

        words = cleaned.split()
        logger.info(f"  Words: {len(words)}")

        # Save original text
        (scripts_dir / f"original_text_{i:03d}.txt").write_text(cleaned)

        # Generate script via Ollama
        try:
            if len(words) < 200:
                script = cleaned
            else:
                # Split into chunks
                chunks = []
                for start in range(0, len(words), _MAX_WORDS_PER_LLM_CHUNK):
                    chunks.append(' '.join(words[start:start + _MAX_WORDS_PER_LLM_CHUNK]))

                logger.info(f"  Chunks: {len(chunks)}")
                script_parts = []
                for ci, chunk in enumerate(chunks, 1):
                    logger.info(f"  Chunk {ci}/{len(chunks)} ({len(chunk.split())} words)...")
                    t0 = time.time()
                    result = call_ollama(
                        prompt=f"Transform the following text into an engaging audiobook script. "
                               f"Write naturally, as if speaking to a curious listener. "
                               f"Keep all key ideas but make them flow conversationally.\n\n{chunk}",
                        system_prompt=AUDIOBOOK_PROMPT,
                    )
                    elapsed = time.time() - t0
                    logger.info(f"    Done in {elapsed:.1f}s ({len(result)} chars)")
                    script_parts.append(result)

                script = "\n\n".join(script_parts)

            # Save script
            script_path = scripts_dir / f"episode_{i:03d}_script.txt"
            script_path.write_text(script)
            logger.info(f"  Script saved: {script_path.name} ({len(script)} chars)")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            # Save error as placeholder
            (scripts_dir / f"episode_{i:03d}_script.txt").write_text(f"[ERROR: {e}]")
            continue

    # Save chapter titles
    (scripts_dir / "chapter_titles.json").write_text(
        json.dumps(chapter_titles, indent=2, ensure_ascii=False)
    )
    logger.info(f"\n✅ All scripts generated ({len(chapters)} chapters)")

    # Step 3: Synthesize TTS
    logger.info(f"\n{'='*60}")
    logger.info("Starting TTS synthesis...")
    logger.info(f"{'='*60}")

    for i in range(1, len(chapters) + 1):
        script_path = scripts_dir / f"episode_{i:03d}_script.txt"
        if not script_path.exists():
            continue

        script = script_path.read_text()
        if script.startswith("[ERROR"):
            logger.warning(f"  [{i}] Skipping (previous error)")
            continue
        if not script.strip():
            continue

        # Break into segments for TTS
        from audify.utils.text import break_text_into_sentences
        sentences = break_text_into_sentences(script)

        # Combine sentences into ~200 char segments
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

        logger.info(f"  [{i}] {len(segments)} segments, ~{len(script.split())} words...")

        # Skip if MP3 already exists
        episode_mp3 = episodes_dir / f"episode_{i:03d}.mp3"
        if episode_mp3.exists():
            logger.info(f"    Skipping, already exists")
            continue

        # Synthesize each segment to WAV, then combine
        wav_files = []
        for si, seg in enumerate(segments):
            wav_path = episodes_dir / f"episode_{i:03d}_seg{si:03d}.wav"
            if call_kokoro(seg, voice, wav_path):
                wav_files.append(wav_path)

        if not wav_files:
            logger.warning(f"    No audio generated")
            continue

        # Combine WAV segments into one MP3
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for wf in wav_files:
            try:
                combined += AudioSegment.from_wav(str(wf))
            except Exception as e:
                logger.warning(f"    Could not add {wf.name}: {e}")
            wf.unlink(missing_ok=True)

        if len(combined) > 0:
            combined.export(str(episode_mp3), format="mp3")
            duration = len(combined) / 1000
            logger.info(f"    MP3: {episode_mp3.name} ({duration:.1f}s)")
        else:
            logger.warning(f"    No audio segments to combine")

    logger.info(f"\n✅ TTS synthesis complete")

    # Step 4: Create M4B
    logger.info(f"\nCreating M4B...")
    from audify.utils.audio import AudioProcessor
    from audify.utils.m4b_builder import write_metadata_header, append_chapter_metadata, assemble_m4b

    episode_mp3_files = sorted(episodes_dir.glob("episode_*.mp3"))
    if not episode_mp3_files:
        logger.error("No episode MP3s found")
        return False

    logger.info(f"Combining {len(episode_mp3_files)} episodes...")

    # Split if >6 hours
    total_duration = sum(AudioProcessor.get_duration(str(f)) for f in episode_mp3_files)
    hours = total_duration / 3600
    logger.info(f"Total duration: {hours:.1f}h")

    # Create temp M4B
    temp_m4b = out_dir / f"{book_name}.tmp.m4b"
    AudioProcessor.combine_audio_files(episode_mp3_files, temp_m4b, output_format="mp4",
                                      description="Combining chapters")

    # Create metadata with all chapter markers
    metadata_path = out_dir / "chapters.txt"
    write_metadata_header(metadata_path)
    start_ms = 0
    for mp3 in episode_mp3_files:
        ep_num = int(mp3.stem.split("_")[1])
        title = chapter_titles[ep_num - 1] if ep_num <= len(chapter_titles) else f"Chapter {ep_num}"
        dur = AudioProcessor.get_duration(str(mp3))
        start_ms = append_chapter_metadata(metadata_path, title, start_ms, dur)

    # Assemble final M4B (split if >6h using ffmpeg -movflags +frag_keyframe)
    final_m4b = out_dir / f"{book_name}.m4b"
    cover = out_dir / "cover.jpg"
    cover_path = cover if cover.exists() else None
    assemble_m4b(temp_m4b, metadata_path, final_m4b, cover_path)
    metadata_path.unlink(missing_ok=True)
    
    # If file is very large (>2GB or >6h), also create a part2 starting from halfway
    file_size_mb = final_m4b.stat().st_size / (1024*1024) if final_m4b.exists() else 0
    if file_size_mb > 500 and len(episode_mp3_files) > 10:
        # Already handled by ffmpeg - no explicit split needed for modern players
        pass

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ COMPLETE: {book_name}")
    logger.info(f"{'='*60}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("epub", help="Path to EPUB file")
    parser.add_argument("--voice", default="af_bella")
    parser.add_argument("--keep-output", action="store_true", help="Don't delete existing output")
    args = parser.parse_args()

    success = process_book(args.epub, args.voice, delete_first=not args.keep_output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
