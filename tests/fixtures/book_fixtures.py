import tempfile
from pathlib import Path

import pytest
from ebooklib import epub
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def book(tmp_dir):
    book = epub.EpubBook()
    # Initialize toc if not already set
    if not hasattr(book, "toc") or book.toc is None:
        book.toc = []

    # Set toc to be the first item in the spine
    book.spine = ["nav"] + list(book.toc)
    nav = b"""<?xml version='1.0' encoding='utf-8'?>
    <!DOCTYPE html>
    <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
    <head>
        <title>Table of Contents</title>
    </head>
    <body>
        <nav id="toc" epub:type="toc">
            <ol>
                <li><a href="chapter1.xhtml">Chapter 1</a></li>
                <li><a href="chapter2.xhtml">Chapter 2</a></li>
            </ol>
        </nav>
    </body>
    </html>"""
    book.set_title("Test Book")
    book.add_metadata("DC", "language", "en")
    book.add_metadata("spine", "toc", nav)
    book.add_author("Test Author")
    book.add_item(
        epub.EpubHtml(
            title="Chapter 1",
            file_name="chapter1.xhtml",
            content="This is the first chapter.",
        )
    )
    book.add_item(
        epub.EpubHtml(
            title="Chapter 2",
            file_name="chapter2.xhtml",
            content="This is the second chapter.",
        )
    )
    # add cover image
    cover_image_path = tmp_dir / "cover.jpg"
    # create a cover blank image file to be used as cover image
    cover_image = Image.new("RGB", (600, 800), "white")
    cover_image.save(cover_image_path)
    book_path = tmp_dir / "test_book.epub"
    cover_image_content = open(cover_image_path, "rb").read()
    epub.EpubItem(uid="cover", file_name="cover.jpg", content=cover_image_content)
    epub.write_epub(book_path, book, {})
    # save the book in a tmp directory
    book_path = tmp_dir / "test_book.epub"
    epub.write_epub(str(book_path), book, {})
    return book
