from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydub.exceptions import CouldntDecodeError

from audify.utils import (
    break_text_into_sentences,
    break_too_long_sentences,
    clean_text,
    combine_small_sentences,
    get_audio_duration,
    get_file_extension,
    get_file_name_title,
)

MODULE_PATH = Path(__file__).resolve().parents[1]


@pytest.fixture
def pdf_file_create(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("This is a test PDF file.")
    return pdf_path


def test_text_is_cleaned(readers):
    for reader in readers:
        cleaned_text = clean_text(reader.get_cleaned_text())
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


def test_break_too_long_sentences():
    sentences = [
        "This is a very long sentence that needs to be broken into smaller parts."
    ]
    expected = [
        "This is a very long sentence that needs to be",
        "broken into smaller parts.",
    ]
    assert break_too_long_sentences(sentences, max_length=50) == expected


def test_break_text_into_sentences():
    text = "This is a test. This is another test sentence."
    expected = ["This is a test.", "This is another test sentence."]
    assert break_text_into_sentences(text, max_length=50, min_length=10) == expected


@patch("audify.utils.AudioSegment.from_file")
def test_get_audio_duration(mock_from_file):
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = 10000  # 10 seconds
    mock_from_file.return_value = mock_audio
    assert get_audio_duration("test.wav") == 10.0


def test_get_audio_duration_culdnt_decode_error():
    with patch("audify.utils.AudioSegment.from_file", side_effect=CouldntDecodeError):
        assert get_audio_duration("test.wav") == 0.0


def test_get_file_extension():
    assert get_file_extension("test.wav") == ".wav"


def test_get_file_name_title():
    title = "This is a Test Title!"
    expected = "this_is_a_test_title"
    assert get_file_name_title(title) == expected
