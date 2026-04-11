#!/usr/bin/env python3
"""Test new sentence splitting logic."""

import re


def normalize_chinese_punctuation(text: str) -> str:
    """Add a space after Chinese punctuation marks if not already followed by whitespace."""
    # Chinese punctuation marks: 。！？；：
    # Replace each punctuation mark with mark + space, but only if not already followed by whitespace
    # Use lookahead to avoid double spaces
    for punct in "。！？；：":
        # regex: punct followed by non-whitespace and not end-of-string
        text = re.sub(f"({punct})(?=\\S)", f"\\1 ", text)
    return text


def split_sentences_regex(text: str) -> list[str]:
    """Split text into sentences using punctuation marks."""
    # Original regex: r"(?<=[.!?;:¿¡]) +"
    # New regex: include Chinese punctuation and optional whitespace
    # Split on punctuation followed by whitespace or end-of-string
    # Using lookbehind to keep punctuation with preceding sentence
    return re.split(r"(?<=[.!?;:¿¡。！？；：])\s*", text)


def split_sentences_normalized(text: str) -> list[str]:
    """Normalize Chinese punctuation then split with original regex."""
    normalized = normalize_chinese_punctuation(text)
    return re.split(r"(?<=[.!?;:¿¡]) +", normalized)


# Test cases
test_cases = [
    ("这是第一句话。这是第二句话！", ["这是第一句话。", "这是第二句话！"]),
    ("Hello world. This is a test.", ["Hello world.", "This is a test."]),
    ("Mixed. 这是中文。English!", ["Mixed.", "这是中文。", "English!"]),
    ("No punctuation here", ["No punctuation here"]),
    (
        "Long sentence with commas， but no period。Another sentence.",
        ["Long sentence with commas， but no period。", "Another sentence."],
    ),
]

print("Testing normalize_chinese_punctuation:")
for punct in "。！？；：":
    test = f"测试{punct}测试"
    norm = normalize_chinese_punctuation(test)
    print(f"  '{test}' -> '{norm}'")

print("\nTesting split_sentences_regex:")
for text, expected in test_cases:
    result = split_sentences_regex(text)
    print(f"  Input: {text}")
    print(f"  Result: {result}")
    print(f"  Expected: {expected}")
    print(f"  Match: {result == expected}")

print("\nTesting split_sentences_normalized:")
for text, expected in test_cases:
    result = split_sentences_normalized(text)
    print(f"  Input: {text}")
    print(f"  Result: {result}")
    print(f"  Expected: {expected}")
    print(f"  Match: {result == expected}")

# Test with real chapter text
print("\n--- Testing with real chapter text ---")
import sys

sys.path.insert(0, ".")
from audify.readers.ebook import EpubReader

epub_path = "/home/rd24/Downloads/Concise History of Reform and Opening-up_zh.epub"
reader = EpubReader(epub_path)
chapters = reader.get_chapters()
chapter = chapters[0]
text = reader.extract_text(chapter)[:1000]  # first 1000 chars
print(f"Text sample (first 1000 chars): {text[:100]}...")

# Test regex split
sentences = split_sentences_regex(text)
print(f"Regex split: {len(sentences)} sentences")
for i, s in enumerate(sentences[:5]):
    print(f"  {i}: len={len(s)}: {s[:50]}...")

# Test normalized split
sentences2 = split_sentences_normalized(text)
print(f"Normalized split: {len(sentences2)} sentences")
for i, s in enumerate(sentences2[:5]):
    print(f"  {i}: len={len(s)}: {s[:50]}...")

# Compare with original break_text_into_sentences
from audify.utils.text import break_text_into_sentences

sentences3 = break_text_into_sentences(text, max_length=2000, min_length=20)
print(f"Original function: {len(sentences3)} sentences")
for i, s in enumerate(sentences3[:5]):
    print(f"  {i}: len={len(s)}: {s[:50]}...")
