from pathlib import Path

import pytest

from audify.pdf_read import PdfReader
from audify.utils import break_text_into_sentences, clean_text

MODULE_PATH = Path(__file__).resolve().parents[1]


@pytest.fixture
def pdf_path():
    return MODULE_PATH / "data" / "test.pdf"


@pytest.fixture
def reader(pdf_path):
    return PdfReader(pdf_path)


def test_text_is_cleaned(reader):
    cleaned_text = clean_text(reader.get_cleaned_text())
    assert cleaned_text
    assert ".." not in cleaned_text
    assert "  " not in cleaned_text
    assert "\n" not in cleaned_text
    assert "//" not in cleaned_text


def test_break_text_into_sentences(reader):
    max_length = 239
    min_length = 10
    sentences = break_text_into_sentences(
        reader.get_cleaned_text(), max_length, min_length
    )
    assert sentences
    # Check that all sentences are shorter than 239 characters
    assert all([len(sentence) < max_length for sentence in sentences])
    # Check that all sentences are longer than 10 characters
    assert all([len(sentence) > min_length for sentence in sentences])
