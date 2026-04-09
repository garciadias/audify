import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audify.utils.audio import AudioProcessor
from audify.utils.file_utils import PathManager


def contains_cjk(text: str) -> bool:
    """Check if text contains CJK (Chinese, Japanese, Korean) characters."""
    # Basic CJK Unicode ranges
    for char in text:
        if any(
            start <= ord(char) <= end
            for start, end in [
                (0x4E00, 0x9FFF),  # CJK Unified Ideographs
                (0x3400, 0x4DBF),  # CJK Extension A
                (0x20000, 0x2A6DF),  # CJK Extension B
                (0x2A700, 0x2B73F),  # CJK Extension C
                (0x2B740, 0x2B81F),  # CJK Extension D
                (0x2B820, 0x2CEAF),  # CJK Extension E
                (0x2CEB0, 0x2EBEF),  # CJK Extension F
                (0x3000, 0x303F),  # CJK Symbols and Punctuation
                (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
            ]
        ):
            return True
    return False


def clean_text(text: str) -> str:
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Replace @ with 'a' to avoid TTS errors
    cleaned = cleaned.replace("@", "a")
    # Remove multiple spaces
    cleaned = re.sub(r" +", " ", cleaned)
    # Remove leading and trailing spaces
    cleaned = cleaned.strip()
    # Remove spaces before punctuation, commas, brackets, quotes, and hyphens
    cleaned = re.sub(r" ([.,!?;:¿¡-])", r"\1", cleaned)
    # Removes multiple spaces, tabs, newlines, punctuation and brackets
    cleaned = re.sub(r"[\s\[\]{}()<>/\\#]", " ", cleaned)
    # Remove extra spaces
    cleaned = re.sub(r" +", " ", cleaned)
    # Remove multiple punctuation marks
    cleaned = re.sub(r"([.,!?;:¿¡-])+", r"\1", cleaned)
    # remove * and _ characters
    cleaned = cleaned.replace("*", "").replace("_", "")
    return cleaned


def combine_small_sentences(sentences: list[str], min_length: int = 10) -> list[str]:
    result: list[str] = []
    for sentence in sentences:
        if len(sentence) < min_length:
            if result:
                result[-1] += " " + sentence
        else:
            result.append(sentence)
    return result


def break_too_long_sentences(sentences: list[str], max_length: int = 239) -> list[str]:
    result: list[str] = []
    for sentence in sentences:
        # If sentence contains CJK characters and has few spaces, split by character length
        if (
            contains_cjk(sentence) and sentence.count(" ") < len(sentence) / 50
        ):  # heuristic
            # Split by character count
            start = 0
            while start < len(sentence):
                # Take up to max_length characters, but try to break at punctuation if possible
                end = start + max_length
                if end >= len(sentence):
                    result.append(sentence[start:].strip())
                    break
                # Look for punctuation break in the last 20% of segment
                lookback = int(max_length * 0.2)
                punctuation_marks = "。！？；：.!?;:¿¡"
                break_pos = -1
                for i in range(end - 1, end - lookback - 1, -1):
                    if i >= start and sentence[i] in punctuation_marks:
                        break_pos = i + 1  # include punctuation
                        break
                if break_pos > start:
                    result.append(sentence[start:break_pos].strip())
                    start = break_pos
                else:
                    # No punctuation found, split at max_length
                    result.append(sentence[start:end].strip())
                    start = end
        else:
            # Original word-based splitting for non-CJK or spaced text
            sentence_words = sentence.split()
            new_sentence = ""
            for word in sentence_words:
                if len(new_sentence) + len(word) > max_length:
                    result.append(new_sentence.strip(" "))
                    new_sentence = ""
                new_sentence += word + " "
            if new_sentence:
                result.append(new_sentence.strip(" "))
    return result


def break_text_into_sentences(
    text: str, max_length: int = 5000, min_length: int = 20
) -> list[str]:
    # Split text into sentences using punctuation marks (Western and Chinese)
    # Includes: .!?;:¿¡。！？；：
    # Split on punctuation followed by optional whitespace
    sentences = re.split(r"(?<=[.!?;:¿¡。！？；：])\s*", text)
    # Filter out empty strings
    sentences = [s for s in sentences if s.strip()]
    # Split long sentences into smaller ones to avoid TTS errors
    result = break_too_long_sentences(sentences, max_length - min_length)
    # Combine sentences that are too short with the previous one
    result = combine_small_sentences(result, min_length)
    # Parallelize the cleaning of the sentences
    with ThreadPoolExecutor() as executor:
        result = list(executor.map(clean_text, result))
    return result


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds. Delegates to AudioProcessor.get_duration."""
    return AudioProcessor.get_duration(file_path)


def get_file_extension(file_path: str) -> str:
    return Path(file_path).suffix


def get_file_name_title(title: str) -> str:
    """Convert a title to a filesystem-safe snake_case name.

    Delegates to ``PathManager.clean_file_name`` so the logic lives in one place.
    """
    return PathManager.clean_file_name(title)
