"""Tests for PDF as Source ebook: metadata, Book key and publishing."""

from pathlib import Path
from typing import Any, Optional

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


def test_title_from_filename_keeps_acronyms_and_mixed_case():
    assert title_from_filename(Path("intro_to_NLP.pdf")) == "Intro To NLP"
    assert title_from_filename(Path("iPhone_Guide.pdf")) == "iPhone Guide"


def _pypdf_file(path: Path, info: dict[str, Any]) -> Path:
    """A PDF whose info entries are written raw (null and indirect values
    included); add_metadata would turn them into text."""
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, TextStringObject

    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    entries: Any = writer._info.get_object() if writer._info else None
    assert entries is not None
    for key, value in info.items():
        if isinstance(value, str):
            value = TextStringObject(value)
        entries[NameObject(key)] = value
    writer.write(path)
    return path


def test_null_info_values_are_absent(tmp_path):
    from pypdf.generic import NullObject

    path = _pypdf_file(
        tmp_path / "null_title.pdf", {"/Title": NullObject(), "/Author": NullObject()}
    )
    meta = book_metadata_from_pdf(path)
    assert meta.title == "Null Title"
    assert meta.authors == []


def test_indirect_title_is_resolved(tmp_path):
    from pypdf.generic import IndirectObject, TextStringObject

    def build(path: Path) -> Path:
        from pypdf import PdfWriter
        from pypdf.generic import NameObject

        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        ref: IndirectObject = writer._add_object(TextStringObject("Indirect Title"))
        entries: Any = writer._info.get_object() if writer._info else None
        entries[NameObject("/Title")] = ref
        writer.write(path)
        return path

    path = build(tmp_path / "x.pdf")
    assert book_metadata_from_pdf(path).title == "Indirect Title"


def test_author_separator_is_case_insensitive(tmp_path):
    path = _pypdf_file(tmp_path / "x.pdf", {"/Author": "Jane Doe AND John Smith"})
    assert book_metadata_from_pdf(path).authors == ["Jane Doe", "John Smith"]


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
