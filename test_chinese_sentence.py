#!/usr/bin/env python3
"""Test Chinese sentence splitting."""

import sys

sys.path.insert(0, ".")

from audify.utils.text import break_text_into_sentences, clean_text

# Sample Chinese text with punctuation
chinese_text = "这是第一句话。这是第二句话！这是第三句话？这是第四句话；这是第五句话：还有第六句话。"

print("Original text:", chinese_text)
print("\nCurrent split:")
sentences = break_text_into_sentences(chinese_text, max_length=100, min_length=5)
for i, s in enumerate(sentences):
    print(f"{i}: {s} (len={len(s)})")

# Test with mixed Chinese and Western punctuation
mixed = "这是第一句话. This is second sentence. 这是第三句话！"
print("\nMixed text:", mixed)
sentences2 = break_text_into_sentences(mixed, max_length=100, min_length=5)
for i, s in enumerate(sentences2):
    print(f"{i}: {s} (len={len(s)})")

# Test with long Chinese sentence without punctuation
long_chinese = "这是一个非常长的中文句子，没有句号分隔，只有逗号，因此整个句子可能会非常长，导致Google TTS报错，因为超过了最大字符限制。"
print("\nLong Chinese with only commas:", long_chinese)
sentences3 = break_text_into_sentences(long_chinese, max_length=50, min_length=5)
for i, s in enumerate(sentences3):
    print(f"{i}: {s} (len={len(s)})")
