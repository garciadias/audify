import os
import tempfile
from pathlib import Path

import pytest
from ebooklib import epub
from reportlab.pdfgen import canvas

from audify.readers.pdf import PdfReader

MODULE_PATH = Path(__file__).resolve().parents[2]


@pytest.fixture
def pdfs() -> list[str]:
    # PDF file in english
    tmp_folder = tempfile.tempdir
    path_en = f"{tmp_folder}/test_en.pdf"
    c = canvas.Canvas(path_en, pagesize=(8.27 * 72, 11.7 * 72))
    c.drawString(100, 700, "This is a test PDF content.")
    c.save()
    path_pt = f"{tmp_folder}/test_pt.pdf"
    c = canvas.Canvas(path_pt, pagesize=(8.27 * 72, 11.7 * 72))
    c.drawString(100, 700, "Este é um conteúdo de teste em PDF.")
    c.save()
    return [str(path_en), str(path_pt)]


@pytest.fixture
def readers(pdfs):
    return [PdfReader(file_path) for file_path in pdfs]


@pytest.fixture(scope="session")
def test_pdf_file():
    """Create a simple test PDF file that persists for the session."""
    # Create in current working directory so tests can find it
    pdf_path = "test.pdf"

    # Create a simple PDF with test content
    c = canvas.Canvas(pdf_path, pagesize=(8.27 * 72, 11.7 * 72))
    c.drawString(100, 700, "This is a test PDF content.")
    c.drawString(100, 680, "Multiple sentences for testing.")
    c.drawString(100, 660, "More test content here.")
    c.save()

    yield pdf_path

    # Cleanup after all tests are done
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


@pytest.fixture(scope="session")
def test_epub_file():
    """Create a simple test EPUB file that persists for the session."""
    # Create in current working directory so tests can find it
    epub_path = "test.epub"

    # Create a simple EPUB with test content
    book = epub.EpubBook()
    book.set_identifier('test123')
    book.set_title('Test Book')
    book.set_language('en')
    book.add_author('Test Author')

    # Create a chapter
    chapter = epub.EpubHtml(
        title='Chapter 1', file_name='chapter1.xhtml', lang='en'
    )
    chapter.content = '''<html>
    <head><title>Chapter 1</title></head>
    <body>
    <h1>Chapter 1</h1>
    <p>This is test content for the EPUB file. It contains multiple sentences
    for testing purposes.</p>
    <p>More test content here to ensure proper text extraction.</p>
    </body>
    </html>'''

    book.add_item(chapter)
    book.toc = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ['nav', chapter]

    # Write the EPUB file
    epub.write_epub(epub_path, book, {})

    yield epub_path

    # Cleanup after all tests are done
    if os.path.exists(epub_path):
        os.remove(epub_path)
