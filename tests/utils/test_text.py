from unittest.mock import patch

import pytest

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
