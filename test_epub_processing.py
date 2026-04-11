#!/usr/bin/env python3
"""Test EPUB processing with the actual file."""

import sys

import pytest

sys.path.insert(0, ".")

from audify.readers.ebook import EpubReader


@pytest.mark.skip(reason="Manual debug script; requires explicit EPUB path")
def test_epub_processing(epub_path):
    print(f"Testing EPUB: {epub_path}")
    reader = EpubReader(epub_path)

    print(f"\nBook title: {reader.title}")

    chapters = reader.get_chapters()
    print(f"\nNumber of chapters found: {len(chapters)}")

    for i, chapter in enumerate(chapters):
        text = reader.extract_text(chapter)
        preview = text[:200].replace("\n", " ")
        print(f"\n--- Chapter {i} ---")
        print(f"Preview: {preview}")

        # Check if this looks like TOC
        lower_text = text.lower()
        toc_indicators = ["table of contents", "contents", "目录", "chapter", "part"]
        has_toc_indicator = any(indicator in lower_text for indicator in toc_indicators)

        # Check if it's mostly a list of titles/links
        lines = text.split("\n")
        short_lines = [line for line in lines if 10 < len(line.strip()) < 100]
        if has_toc_indicator and len(short_lines) > 3:
            print(
                f"*** WARNING: This looks like a TOC file! (has {len(short_lines)} short lines)"
            )

    return reader


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_epub_processing.py <epub_file>")
        sys.exit(1)

    epub_path = sys.argv[1]
    test_epub_processing(epub_path)
