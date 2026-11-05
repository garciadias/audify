"""Tests for the per-book Metadata override and field provenance."""

import json
import logging
from pathlib import Path

from ebooklib import epub

from audify.library.metadata import BookMetadata, SeriesEntry, book_metadata_from_epub
from audify.library.override import OVERRIDE_FILE, apply_override, load_override
from audify.library.publish import abs_metadata


def _ebook_meta() -> BookMetadata:
    meta = BookMetadata(
        title="The Capital Order",
        authors=["Clara E. Mattei"],
        language="en",
        publisher="UChicago",
    )
    meta.mark("ebook")
    return meta


def test_override_wins_and_is_marked(tmp_path):
    meta = apply_override(
        _ebook_meta(),
        {"title": "A Ordem do Capital", "language": "pt", "series": ["Ensaios #2"]},
    )
    assert meta.title == "A Ordem do Capital"
    assert meta.language == "pt"
    assert meta.series == [SeriesEntry("Ensaios", "2")]
    assert meta.provenance["title"] == "override"
    assert meta.provenance["series"] == "override"
    # Fields absent from the override fall through to the ebook.
    assert meta.authors == ["Clara E. Mattei"]
    assert meta.provenance["authors"] == "ebook"


def test_override_can_clear_a_wrong_value():
    meta = apply_override(_ebook_meta(), {"publisher": None})
    assert meta.publisher == ""
    assert "publisher" not in meta.provenance


def test_override_accepts_strings_for_list_fields():
    meta = apply_override(_ebook_meta(), {"genres": "History", "authors": ["A", ""]})
    assert meta.genres == ["History"]
    assert meta.authors == ["A"]


def test_load_override_ignores_unknown_fields(tmp_path, caplog):
    path = tmp_path / OVERRIDE_FILE
    path.write_text(json.dumps({"title": "T", "rating": 5}))
    with caplog.at_level(logging.WARNING):
        data = load_override(path)
    assert data == {"title": "T"}
    assert "rating" in caplog.text


def test_load_override_missing_or_broken(tmp_path):
    assert load_override(tmp_path / OVERRIDE_FILE) is None
    (tmp_path / OVERRIDE_FILE).write_text("{not json")
    assert load_override(tmp_path / OVERRIDE_FILE) is None


def test_ebook_fields_are_marked_ebook(tmp_path):
    book = epub.EpubBook()
    book.set_identifier("x")
    book.set_title("T")
    book.set_language("en")
    book.add_author("A")
    meta = book_metadata_from_epub(book, narrators=["Kokoro (af_bella)"])
    assert meta.provenance == {
        "title": "ebook",
        "authors": "ebook",
        "language": "ebook",
        "narrators": "audify",
    }


def test_creator_applies_override_from_the_work_directory(tmp_path):
    from audify.audiobook_creator import AudiobookCreator
    from audify.readers.ebook import EpubReader

    book = epub.EpubBook()
    book.set_identifier("x")
    book.set_title("The Capital Order")
    book.set_language("en")
    book.add_author("Clara E. Mattei")
    chapter = epub.EpubHtml(title="One", file_name="one.xhtml", lang="en")
    chapter.content = "<h1>One</h1><p>Text.</p>"
    book.add_item(chapter)
    book.spine = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)

    work = tmp_path / "work" / "the_capital_order"
    work.mkdir(parents=True)
    (work / OVERRIDE_FILE).write_text(
        json.dumps({"title": "A Ordem do Capital", "language": "pt"})
    )

    creator = object.__new__(AudiobookCreator)
    creator.reader = EpubReader(path)
    creator.title = creator.reader.title
    creator.language = "en"
    creator.audiobook_path = work
    creator.tts_provider = "kokoro"
    creator.speaker = "pf_dora"
    creator.task_name = "direct"
    creator._requires_llm = False
    creator.translate = None
    meta = creator._build_book_metadata()

    assert meta.title == "A Ordem do Capital"
    assert meta.language == "pt"
    assert meta.provenance["title"] == "override"
    data = abs_metadata(meta)
    assert data["title"] == "A Ordem do Capital"
    assert data["language"] == "pt"
    assert Path(work / OVERRIDE_FILE).exists()
