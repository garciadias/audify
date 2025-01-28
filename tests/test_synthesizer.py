from unittest.mock import patch

import pytest
from ebooklib import epub

from audify.synthesizer import BookSynthesizer


@pytest.fixture
def book():
    book = epub.EpubBook()
    book.set_title("Test Book")
    book.add_metadata("DC", "language", "en")
    return book


@pytest.fixture
def synthesizer(book):
    with patch("audify.ebook_read.save_book_cover_image", return_value="cover.jpg"):
        return BookSynthesizer(book)


def test_initialize_metadata(synthesizer):
    synthesizer._initialize_metadata()
    metadata_file = synthesizer.audio_book_path / "chapters.txt"
    assert metadata_file.exists()
    with open(metadata_file, "r") as f:
        content = f.read()
    assert "major_brand=M4A" in content


def test_synthesize(synthesizer):
    with (
        patch("audify.text_to_speech.process_chapter") as mock_process_chapter,
        patch("audify.text_to_speech.create_m4b") as mock_create_m4b,
    ):
        synthesizer.synthesize()
        mock_process_chapter.assert_called()
        mock_create_m4b.assert_called()


def test_process_chapters(synthesizer):
    with (
        patch("audify.ebook_read.read_chapters", return_value=["chapter1", "chapter2"]),
        patch(
            "audify.text_to_speech.process_chapter", return_value=1000
        ) as mock_process_chapter,
    ):
        synthesizer._process_chapters()
        assert mock_process_chapter.call_count == 2


def test_create_m4b(synthesizer):
    with patch("audify.text_to_speech.create_m4b") as mock_create_m4b:
        synthesizer._create_m4b()
        mock_create_m4b.assert_called()
