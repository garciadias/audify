#!/usr/bin/env python3
"""Test updated sentence splitting."""

import sys

sys.path.insert(0, ".")

from audify.utils.text import break_text_into_sentences, contains_cjk

# Test contains_cjk
print("Testing contains_cjk:")
assert contains_cjk("中文") == True
assert contains_cjk("Hello") == False
assert contains_cjk("Hello 中文") == True
print("  OK")

# Test basic splitting
print("\nTest basic splitting:")
text = "这是第一句话。这是第二句话！"
sentences = break_text_into_sentences(text, max_length=100, min_length=5)
print(f"  Input: {text}")
print(f"  Sentences: {sentences}")
assert len(sentences) == 2
assert sentences[0] == "这是第一句话。"
assert sentences[1] == "这是第二句话！"
print("  OK")

# Test mixed punctuation
text2 = "Hello world. 这是中文。English!"
sentences2 = break_text_into_sentences(text2, max_length=100, min_length=5)
print(f"\n  Input: {text2}")
print(f"  Sentences: {sentences2}")
assert len(sentences2) == 3
print("  OK")

# Test long Chinese sentence without punctuation (only commas)
long = "这是一个非常长的中文句子，没有句号分隔，只有逗号，因此整个句子可能会非常长，导致Google TTS报错，因为超过了最大字符限制。"
sentences3 = break_text_into_sentences(long, max_length=50, min_length=5)
print(f"\n  Input: {long}")
print(f"  Sentences: {len(sentences3)}")
for i, s in enumerate(sentences3):
    print(f"    {i}: len={len(s)}: {s}")
# Expect split by character length because contains CJK and few spaces
assert len(sentences3) > 1
print("  OK")

# Test with real chapter
print("\n--- Testing with real chapter ---")
from audify.readers.ebook import EpubReader

epub_path = "/home/rd24/Downloads/Concise History of Reform and Opening-up_zh.epub"
reader = EpubReader(epub_path)
chapters = reader.get_chapters()
chapter = chapters[0]
text = reader.extract_text(chapter)
print(f"Chapter length: {len(text)}")

sentences = break_text_into_sentences(text, max_length=2000, min_length=20)
print(f"Number of sentences: {len(sentences)}")
print("Sentence lengths:")
for i, s in enumerate(sentences[:10]):
    print(f"  {i}: len={len(s)}")
if len(sentences) > 10:
    print(f"  ... and {len(sentences) - 10} more")

# Check for overly long sentences
long_sentences = [s for s in sentences if len(s) > 1000]
print(f"Sentences longer than 1000 chars: {len(long_sentences)}")
if long_sentences:
    print("  WARNING: Some sentences still too long!")

# Ensure no empty sentences
empty = [s for s in sentences if not s.strip()]
assert len(empty) == 0, f"Empty sentences: {empty}"
print("  No empty sentences.")

print("\nAll tests passed!")
