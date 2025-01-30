from unittest.mock import MagicMock, patch

import pytest

from audify.synthesizer import BookSynthesizer


@pytest.fixture
def synthesizer(tmp_dir, book):
    with patch("audify.ebook_read.epub.read_epub", return_value=book):
        return BookSynthesizer(book_path=tmp_dir / "test_book.epub")


def test_initialize_metadata(synthesizer):
    """Test the initialization of metadata in the synthesizer."""
    synthesizer.synthesize()
    metadata_file = synthesizer.audio_book_path / "chapters.txt"
    with open(metadata_file, "r") as metadata_file_handle:
        content = metadata_file_handle.read()
    assert content == "Chapter 1\nChapter 2\n"


def test_synthesize(synthesizer):
    with patch(
        "audify.synthesizer.BookSynthesizer._process_chapter"
    ) as mock_process_chapter:
        synthesizer.synthesize()
        assert mock_process_chapter.called


def test_synthesize_processes_chapters(synthesizer):
    with (
        patch(
            "audify.ebook_read.BookReader.read_chapters",
            return_value=[
                MagicMock(sentences=["test sentence 1"]),
                MagicMock(sentences=["test sentence 2"]),
            ],
        ),
        patch(
            "audify.synthesizer.BookSynthesizer._process_chapter", return_value=1000
        ) as mock_process_chapter,
    ):
        synthesizer.synthesize()
        assert mock_process_chapter.call_count == 2


def test_create_m4b(synthesizer):
    with patch("audify.synthesizer.BookSynthesizer._create_m4b") as mock_create_m4b:
        synthesizer._create_m4b([], "dummy.epub", cover_image_path=None)
        mock_create_m4b.assert_called()
