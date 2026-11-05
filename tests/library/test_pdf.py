"""Tests for PDF as Source ebook: metadata, Book key and publishing."""

from pathlib import Path
from typing import Optional

from reportlab.pdfgen import canvas

from audify.audiobook_creator import AudiobookCreator
from audify.library.identity import ItemIdentity
from audify.library.metadata import book_metadata_from_pdf, title_from_filename
from audify.library.publish import find_library_item, publish_library_item
from audify.readers.pdf import PdfReader


def _pdf(
    path: Path,
    text: str = "Chapter one. It was a dark night.",
    title: Optional[str] = None,
    author: Optional[str] = None,
    subject: Optional[str] = None,
    keywords: Optional[str] = None,
) -> Path:
    c = canvas.Canvas(str(path))
    if title is not None:
        c.setTitle(title)
    if author is not None:
        c.setAuthor(author)
    if subject is not None:
        c.setSubject(subject)
    if keywords is not None:
        c.setKeywords(keywords)
    c.drawString(72, 720, text)
    c.save()
    return path


def test_info_dictionary_becomes_book_metadata(tmp_path):
    path = _pdf(
        tmp_path / "x.pdf",
        title="MLOps with Databricks",
        author="Maria Vechtomova and Basak Eskili",
        subject="Production ML on Databricks.",
        keywords="mlops, databricks",
    )
    meta = book_metadata_from_pdf(path, narrators=["Kokoro (af_bella)"])
    assert meta.title == "MLOps with Databricks"
    assert meta.authors == ["Maria Vechtomova", "Basak Eskili"]
    assert meta.description == "Production ML on Databricks."
    assert meta.tags == ["mlops", "databricks"]
    assert meta.provenance["title"] == "ebook"
    assert meta.provenance["narrators"] == "audify"


def test_empty_info_dictionary_uses_the_file_name(tmp_path):
    path = _pdf(tmp_path / "the_wandering-earth.pdf", title="", author="")
    meta = book_metadata_from_pdf(path)
    assert meta.title == "The Wandering Earth"
    assert meta.provenance["title"] == "filename"
    assert meta.authors == []
    assert meta.isbn == ""


def test_word_processor_titles_are_ignored(tmp_path):
    for junk in ["Microsoft Word - draft7.docx", "thesis_final.pdf"]:
        meta = book_metadata_from_pdf(_pdf(tmp_path / "real_name.pdf", title=junk))
        assert meta.title == "Real Name"


def test_generator_placeholders_are_ignored(tmp_path):
    # reportlab's defaults when no title or author is set.
    meta = book_metadata_from_pdf(_pdf(tmp_path / "mlops_with_databricks.pdf"))
    assert meta.title == "Mlops With Databricks"
    assert meta.authors == []


def test_unreadable_pdf_still_gets_a_title(tmp_path):
    path = tmp_path / "broken_book.pdf"
    path.write_bytes(b"not a pdf")
    assert book_metadata_from_pdf(path).title == "Broken Book"


def test_title_from_filename_keeps_acronyms():
    assert title_from_filename(Path("intro_to_NLP.pdf")) == "Intro To NLP"


def _creator(pdf: Path) -> AudiobookCreator:
    creator = object.__new__(AudiobookCreator)
    creator.reader = PdfReader(pdf)
    creator.path = pdf
    creator.task_name = "direct"
    creator.translate = None
    return creator


def test_rerunning_the_same_pdf_finds_the_same_item(tmp_path):
    root = tmp_path / "lib"
    pdf = _pdf(tmp_path / "book.pdf", title="A Book", author="Ann Author")
    audio = tmp_path / "a.m4b"
    audio.write_bytes(b"audio")

    identity = _creator(pdf)._item_identity()
    assert identity is not None
    item = publish_library_item(
        root, book_metadata_from_pdf(pdf), [audio], source_ebook=pdf, identity=identity
    )
    assert (item / "book.pdf").exists()

    # Same text, different metadata (a fixed-up title) -> same Book key.
    edited = _pdf(tmp_path / "edited.pdf", title="A Better Title", author="Ann Author")
    again = _creator(edited)._item_identity()
    assert again is not None
    assert again == identity
    assert find_library_item(root, again) == item

    other = _pdf(tmp_path / "other.pdf", text="A different book entirely.")
    assert _creator(other)._item_identity() != identity
    assert find_library_item(root, ItemIdentity("v1:none")) is None
