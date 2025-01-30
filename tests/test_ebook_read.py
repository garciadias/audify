import tempfile
from pathlib import Path

import pytest

from audify.ebook_read import BookReader

MODULE_PATH = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def book_reader(book):
    return BookReader(book)


def test_read_chapters(book_reader):
    chapters = book_reader.read_chapters()
    assert len(chapters) == 2
    assert "This is the first chapter." in chapters[0].sentences[0]


def test_break_text_into_sentences(book_reader):
    text = "This is the first sentence. This is the second sentence."
    sentences = book_reader.break_text_into_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "This is the first sentence."
    assert sentences[1] == "This is the second sentence."


def test_get_chapter_title(book_reader):
    chapter = "<h1>Chapter 1</h1><p>This is the first chapter.</p>"
    title = book_reader.get_chapter_title(chapter)
    assert title == "Chapter 1"


def test_get_language(book_reader):
    language = book_reader.get_language()
    assert language == "en"
