from unittest.mock import MagicMock, patch

import pytest
from pydub.exceptions import CouldntDecodeError

from audify.utils.text import (
    break_text_into_sentences,
    break_too_long_sentences,
    clean_text,
    combine_small_sentences,
    get_audio_duration,
    get_file_extension,
    get_file_name_title,
)


@pytest.fixture
def pdf_file_create(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("This is a test PDF file.")
    return pdf_path


def test_text_is_cleaned(readers):
    for reader in readers:
        cleaned_text = clean_text(reader.read())
        assert cleaned_text
        assert ".." not in cleaned_text
        assert "  " not in cleaned_text
        assert "\n" not in cleaned_text
        assert "//" not in cleaned_text


def test_clean_text():
    text = " This is a test text.  With  multiple spaces, tabs\t, and newlines\n."
    expected = "This is a test text. With multiple spaces, tabs, and newlines."
    assert clean_text(text) == expected


def test_combine_small_sentences():
    sentences = ["This is a test.", "Short.", "Another test sentence."]
    expected = ["This is a test. Short.", "Another test sentence."]
    assert combine_small_sentences(sentences, min_length=10) == expected


def test_combine_small_sentences_respects_max_length():
    """Test that combine_small_sentences doesn't exceed max_length."""
    # Create a sentence that's just under max_length and short sentences
    long_sentence = "a" * 40  # 40 chars
    short_sentence1 = "b" * 8  # 8 chars (< min_length=10)
    short_sentence2 = "c" * 8  # 8 chars (< min_length=10)

    sentences = [long_sentence, short_sentence1, short_sentence2]
    # With max_length=50, first can combine with one short (40+1+8=49)
    # but not both (40+1+8+1+8=58 > 50)
    result = combine_small_sentences(sentences, min_length=10, max_length=50)

    # Should combine first two but not the third
    assert len(result) == 2
    assert result[0] == long_sentence + " " + short_sentence1
    assert result[1] == short_sentence2


def test_break_too_long_sentences():
    sentences = [
        "This is a very long sentence that needs to be broken into smaller parts."
    ]
    expected = [
        "This is a very long sentence that needs to be",
        "broken into smaller parts.",
    ]
    assert break_too_long_sentences(sentences, max_length=50) == expected


def test_break_too_long_sentences_with_long_word():
    """Test that break_too_long_sentences handles words longer than max_length."""
    # Create a sentence with a very long "word" (e.g., URL or technical term)
    long_word = "a" * 100
    sentences = [f"Short text {long_word} more text."]
    result = break_too_long_sentences(sentences, max_length=50)

    # Verify all chunks are within limit
    for chunk in result:
        assert len(chunk) <= 50, f"Chunk exceeds max_length: {len(chunk)} chars"

    # Verify the long word was split
    assert len(result) >= 3  # "Short text", chunks of long_word, "more text."


def test_break_text_into_sentences():
    text = "This is a test. This is another test sentence."
    expected = ["This is a test.", "This is another test sentence."]
    assert break_text_into_sentences(text, max_length=50, min_length=10) == expected


def test_break_text_into_sentences_with_provider_limit():
    """Test that break_text_into_sentences respects provider limits."""
    # Create text that would exceed the limit without proper splitting
    long_word = "word" * 1000  # 4000 chars
    text = f"{long_word}. Short. Another."

    # Simulate OpenAI provider with 4096 limit
    # It should use 90% = 3686 as max_length
    result = break_text_into_sentences(text, tts_provider="openai")

    # Verify no sentence exceeds the provider limit
    for sentence in result:
        msg1 = f"Sentence exceeds API limit: {len(sentence)} chars"
        assert len(sentence) <= 4096, msg1
        # Should actually be under 90% of limit (3686)
        msg2 = f"Sentence exceeds safety limit: {len(sentence)} chars"
        assert len(sentence) <= 3686, msg2


@patch("audify.utils.text.AudioSegment.from_file")
def test_get_audio_duration(mock_from_file):
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = 10000  # 10 seconds
    mock_from_file.return_value = mock_audio
    assert get_audio_duration("test.wav") == 10.0


def test_get_audio_duration_culdnt_decode_error():
    with patch(
        "audify.utils.text.AudioSegment.from_file", side_effect=CouldntDecodeError
    ):
        assert get_audio_duration("test.wav") == 0.0


def test_get_file_extension():
    assert get_file_extension("test.wav") == ".wav"


def test_get_file_name_title():
    title = "This is a Test Title!"
    expected = "this_is_a_test_title"
    assert get_file_name_title(title) == expected
