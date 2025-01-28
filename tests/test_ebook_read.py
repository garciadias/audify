from pathlib import Path

import pytest
from ebooklib import epub

from audify.ebook_read import BookReader

MODULE_PATH = Path(__file__).resolve().parents[1]


@pytest.fixture
def book():
    book = epub.EpubBook()
    book.set_title("Test Book")
    book.add_metadata("DC", "language", "en")
    return book


@pytest.fixture
def synthesizer(book):
    return BookReader(book)


def test_read_chapters(synthesizer):
    item = epub.EpubHtml(title="Chapter s1", file_name="chap_01.xhtml", lang="en")
    item.set_content("<h1>Chapter 1</h1><p>This is the first chapter.</p>")
    synthesizer.book.add_item(item)
    chapters = synthesizer.read_chapters()
    assert len(chapters) == 1
    assert "This is the first chapter." in chapters[0]


def test_extract_text_from_epub_chapter(synthesizer):
    chapter = "<h1>Chapter 1</h1><p>This is the first chapter.</p>"
    text = synthesizer.extract_text_from_epub_chapter(chapter)
    assert text == "Chapter 1\nThis is the first chapter."


def test_break_text_into_sentences(synthesizer):
    text = "This is the first sentence. This is the second sentence."
    sentences = synthesizer.break_text_into_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "This is the first sentence."
    assert sentences[1] == "This is the second sentence."


def test_get_chapter_title(synthesizer):
    chapter = "<h1>Chapter 1</h1><p>This is the first chapter.</p>"
    title = synthesizer.get_chapter_title(chapter)
    assert title == "Chapter 1"


def test_get_book_title(synthesizer):
    title = synthesizer.get_book_title()
    assert title == "test_book"


def test_save_book_cover_image(synthesizer):
    item = epub.EpubImage()
    item.set_content(b"image content")
    item.file_name = "cover.jpg"
    synthesizer.book.add_item(item)
    cover_path = synthesizer.save_book_cover_image()
    assert cover_path == f"{synthesizer.audio_book_path}/cover.jpg"
    assert Path(cover_path).exists()


def test_get_language(synthesizer):
    language = synthesizer.get_language()
    assert language == "en"
