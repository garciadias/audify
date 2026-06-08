#!/usr/bin/env python3
"""
Robust Phase 1: LLM script generation with short per-request timeouts.
Uses direct Ollama API (not LiteLLM) with 30-second read timeout.
Retries failed requests up to 3 times.
Skips chapters with existing scripts.

Usage: uv run python3 scripts/phase1_robust.py <epub_path> [--voice af_bella]
"""

import json
import shutil
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audify.readers.ebook import EpubReader
from audify.utils.text import get_file_name_title
from audify.audiobook_creator import _clean_text_for_audiobook, _MAX_WORDS_PER_LLM_CHUNK

MODEL = "ministral-3:14b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OUTPUT_BASE = Path("data/output")


def call_ollama(prompt: str, system: str = "", timeout: int = 120) -> str:
    """Call Ollama with per-request timeout. Returns '' on failure."""
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL, "prompt": prompt, "system": system,
                "stream": False, "options": {"num_predict": 2000, "temperature": 0.7},
            }, timeout=(30, timeout))  # (connect_timeout, read_timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "")
            if text.strip():
                return text
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
    return ""


def process_book(epub_path: str):
    epub = Path(epub_path)
    if not epub.exists():
        print(f"File not found: {epub}")
        return False

    book_name = get_file_name_title(epub.stem)
    out_dir = OUTPUT_BASE / book_name
    scripts_dir = out_dir / "scripts"

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)
    elif not scripts_dir.exists():
        scripts_dir.mkdir(parents=True, exist_ok=True)

    # Read EPUB
    try:
        reader = EpubReader(str(epub))
        chapters = reader.get_chapters()
    except Exception as e:
        print(f"  READ ERROR: {e}")
        return False

    print(f"  Chapters: {len(chapters)}")

    # Save cover
    try:
        reader.get_cover_image(str(out_dir))
    except Exception:
        pass

    # Generate titles
    chapter_titles = []
    chapter_titles_path = scripts_dir / "chapter_titles.json"
    generate_titles = True

    if chapter_titles_path.exists():
        try:
            chapter_titles = json.loads(chapter_titles_path.read_text())
            generate_titles = (len(chapter_titles) != len(chapters))
        except Exception:
            pass

    if generate_titles:
        chapter_titles = [reader.get_chapter_title(c) or f"Chapter {i+1}" for i, c in enumerate(chapters)]
        chapter_titles_path.write_text(json.dumps(chapter_titles, indent=2))
        print(f"  Titles: {len(chapter_titles)}")

    # Process chapters
    success_count = 0
    for i, chapter_html in enumerate(chapters, 1):
        ep_file = scripts_dir / f"episode_{i:03d}_script.txt"
        if ep_file.exists() and ep_file.stat().st_size > 50:
            success_count += 1
            continue

        # Clean text
        cleaned = _clean_text_for_audiobook(chapter_html)
        words = cleaned.split()
        if not words:
            ep_file.write_text("")
            continue

        # Save original text
        (scripts_dir / f"original_text_{i:03d}.txt").write_text(cleaned)

        # Generate script
        title = chapter_titles[i - 1] if i <= len(chapter_titles) else f"Chapter {i}"
        print(f"  [{i}/{len(chapters)}] {title[:50]}... ({len(words)} words)", end=" ", flush=True)

        try:
            if len(words) < 200:
                result = cleaned
            else:
                # Split into chunks
                chunks = [' '.join(words[j:j+_MAX_WORDS_PER_LLM_CHUNK]) for j in range(0, len(words), _MAX_WORDS_PER_LLM_CHUNK)]
                results = []
                for ci, chunk in enumerate(chunks, 1):
                    t0 = time.time()
                    result = call_ollama(
                        prompt=f"Transform into an engaging audiobook script. Write naturally, conversationally.\n\n{chunk}",
                        timeout=120,
                    )
                    if result:
                        results.append(result)
                        print(f"c{ci}({time.time()-t0:.0f}s)", end=" ", flush=True)
                    else:
                        print(f"c{ci}FAIL", end=" ", flush=True)

                result = "\n\n".join(results) if results else ""

            if result:
                ep_file.write_text(result)
                success_count += 1
                print(f"OK ({len(result)} chars)")
            else:
                ep_file.write_text("[ERROR: empty response]")
                print("EMPTY")
        except Exception as e:
            ep_file.write_text(f"[ERROR: {e}]")
            print(f"ERR: {e}")

    print(f"\n  ✅ {success_count}/{len(chapters)} chapters")
    return success_count > 0


def main():
    if len(sys.argv) < 2:
        epub_paths = sorted(Path("/home/rd24/Downloads/books_deepseek/en").glob("*.epub"))
        epub_paths += sorted(Path("/home/rd24/Downloads/books_deepseek/es").glob("*.epub"))
        epub_paths += sorted(Path("/home/rd24/Downloads/books_deepseek/pt").glob("*.epub"))
    else:
        epub_paths = [Path(sys.argv[1])]

    skip = {"adam_smith_in_beijing", "capital_in_the_twentyfirst_century",
            "chinas_foreign_political_and_economic_relations_an_unconventional_global_power_state__society_in_east_asia"}

    for epub in epub_paths:
        if not epub.exists():
            continue
        book_name = get_file_name_title(epub.stem)
        if book_name in skip:
            print(f"⏭️  {epub.name}")
            continue
        # Check if M4B already exists
        out_dir = OUTPUT_BASE / book_name
        m4b = list(out_dir.glob("*.m4b")) if out_dir.exists() else []
        if m4b:
            print(f"⏭️  {epub.name} (already has M4B)")
            continue

        print(f"\n{'='*60}")
        print(f"{epub.name}")
        print(f"{'='*60}")
        process_book(str(epub))

    print(f"\n✅ All done")


if __name__ == "__main__":
    main()
