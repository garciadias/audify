#!/usr/bin/env python3
"""Test sentence splitting with real Chinese chapter."""

import sys

sys.path.insert(0, ".")

from audify.readers.ebook import EpubReader
from audify.utils.text import break_text_into_sentences

epub_path = "/home/rd24/Downloads/Concise History of Reform and Opening-up_zh.epub"
reader = EpubReader(epub_path)
chapters = reader.get_chapters()
print(f"Total chapters: {len(chapters)}")

# Take first chapter
chapter = chapters[0]
text = reader.extract_text(chapter)
print(f"Chapter text length: {len(text)}")
print("First 500 chars:")
print(text[:500])
print("\nLast 500 chars:")
print(text[-500:])

# Count punctuation
import re

chinese_punct = "。！？；："
western_punct = ".!?;:"
print(f"\nChinese punctuation occurrences:")
for p in chinese_punct:
    count = text.count(p)
    if count:
        print(f"  '{p}': {count}")

print(f"\nWestern punctuation occurrences:")
for p in western_punct:
    count = text.count(p)
    if count:
        print(f"  '{p}': {count}")

# Check for spaces after punctuation
print("\nChecking spaces after Chinese punctuation:")
for p in chinese_punct:
    # Find all positions
    for match in re.finditer(re.escape(p), text):
        pos = match.start()
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            if next_char.isspace():
                print(f"  '{p}' at {pos} followed by whitespace")
            else:
                print(f"  '{p}' at {pos} followed by '{next_char}' (no space)")
        else:
            print(f"  '{p}' at {pos} end of text")

# Now test current sentence splitting
print("\n--- Current sentence splitting ---")
sentences = break_text_into_sentences(text, max_length=2000, min_length=20)
print(f"Number of sentences: {len(sentences)}")
for i, s in enumerate(sentences[:10]):
    print(f"{i}: len={len(s)}: {s[:80]}...")
if len(sentences) > 10:
    print(f"... and {len(sentences) - 10} more")

# Check sentence lengths
long_sentences = [s for s in sentences if len(s) > 1000]
print(f"\nSentences longer than 1000 chars: {len(long_sentences)}")
for i, s in enumerate(long_sentences[:3]):
    print(f"  {i}: len={len(s)}")
