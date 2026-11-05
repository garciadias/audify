"""Tests for rung 2 of Metadata enrichment: LLM inference."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from ebooklib import epub

from audify.audiobook_creator import AudiobookCreator
from audify.library.inference import TEXT_DERIVABLE_FIELDS, infer_text_fields
from audify.library.metadata import METADATA_FIELDS, BookMetadata
from audify.library.override import apply_override

TEXT = "THE GLASS SEA\nA Novel of the North\n\nChapter 1. The ice came early."

REPLY = json.dumps(
    {
        "description": "A fishing village faces an early winter.",
        "genres": ["Fiction", "Literary Fiction", "Fiction"],
        "tags": ["Winter", "fishing villages"],
        "subtitle": "A Novel of the North",
        # A model asked for four fields may still volunteer factual ones.
        "isbn": "9780000000000",
        "authors": ["Invented Person"],
        "publisher": "Made Up Press",
        "published_date": "1999",
        "series": ["Fake #1"],
    }
)


def _gen(reply: str = REPLY):
    calls: list[tuple[str, str]] = []

    def generate(system: str, user: str) -> str:
        calls.append((system, user))
        return reply

    return generate, calls


def test_empty_description_gets_one_with_llm_provenance():
    meta = BookMetadata(title="The Glass Sea")
    filled = infer_text_fields(meta, TEXT, _gen()[0])
    assert meta.description == "A fishing village faces an early winter."
    assert meta.provenance["description"] == "llm"
    assert meta.genres == ["Fiction", "Literary Fiction"]
    assert meta.tags == ["winter", "fishing villages"]
    assert meta.subtitle == "A Novel of the North"
    assert sorted(filled) == sorted(TEXT_DERIVABLE_FIELDS)


def test_no_factual_field_ever_gets_llm_provenance():
    meta = BookMetadata(title="The Glass Sea")
    infer_text_fields(meta, TEXT, _gen()[0])
    factual = set(METADATA_FIELDS) - set(TEXT_DERIVABLE_FIELDS)
    assert not [f for f in factual if meta.provenance.get(f) == "llm"]
    assert meta.isbn == ""
    assert meta.authors == []
    assert meta.publisher == ""
    assert meta.published_date == ""
    assert meta.series == []


def test_existing_values_are_never_overwritten():
    meta = BookMetadata(title="T", description="From the ebook.", genres=["History"])
    meta.mark("ebook")
    meta.subtitle = "From Open Library"
    meta.provenance["subtitle"] = "openlibrary"
    meta = apply_override(meta, {"tags": []})  # cleared on purpose
    generate, calls = _gen()
    assert infer_text_fields(meta, TEXT, generate) == []
    assert calls == []  # nothing to fill, so no LLM call at all
    assert meta.description == "From the ebook."
    assert meta.genres == ["History"]
    assert meta.subtitle == "From Open Library"
    assert meta.tags == []


def test_only_missing_fields_are_filled():
    meta = BookMetadata(title="T", description="Kept.")
    meta.mark("ebook")
    filled = infer_text_fields(meta, TEXT, _gen()[0])
    assert "description" not in filled
    assert meta.description == "Kept."
    assert meta.provenance["description"] == "ebook"


def test_skipped_cleanly_without_an_llm():
    def unreachable(system: str, user: str) -> str:
        raise ConnectionError("no LLM endpoint")

    for generate in [
        unreachable,
        _gen("Error: Could not connect to local LLM server.")[0],
        _gen("")[0],
        _gen("I cannot help with that.")[0],
        _gen("{not json}")[0],
        _gen("[1, 2]")[0],
    ]:
        meta = BookMetadata(title="T")
        assert infer_text_fields(meta, TEXT, generate) == []
        assert meta.provenance == {}


def test_reasoning_preamble_and_bad_values_are_handled():
    reply = '<think>{"x": 1}</think> ' + json.dumps(
        {
            "description": 5,
            "genres": "History",
            "tags": [{"a": 1}],
            "subtitle": "x" * 200,
        }
    )
    meta = BookMetadata(title="T")
    assert infer_text_fields(meta, TEXT, _gen(reply)[0]) == ["genres", "tags"]
    assert meta.genres == ["History"]
    assert meta.description == ""
    assert meta.subtitle == ""


def test_no_text_means_no_call():
    generate, calls = _gen()
    assert infer_text_fields(BookMetadata(title="T"), "   ", generate) == []
    assert calls == []


def _epub(path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("x")
    book.set_title("The Glass Sea")
    book.set_language("en")
    chapter = epub.EpubHtml(title="One", file_name="one.xhtml", lang="en")
    chapter.content = "<h1>One</h1><p>The ice came early.</p>"
    book.add_item(chapter)
    book.spine = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


def test_creator_runs_rung_2_only_when_publishing(tmp_path, monkeypatch):
    from audify.readers.ebook import EpubReader

    monkeypatch.setenv("AUDIFY_METADATA_LOOKUP", "1")
    creator = object.__new__(AudiobookCreator)
    creator.reader = EpubReader(_epub(tmp_path / "b.epub"))
    creator.llm_client = MagicMock()
    creator.llm_client.config.generate.return_value = REPLY
    creator.library_root = tmp_path / "lib"

    meta = BookMetadata(title="The Glass Sea")
    creator._infer_from_text(meta)
    assert meta.provenance["description"] == "llm"
    user_prompt = creator.llm_client.config.generate.call_args.kwargs["user_prompt"]
    assert "The ice came early." in user_prompt

    creator.library_root = None
    creator.llm_client.config.generate.reset_mock()
    creator._infer_from_text(BookMetadata(title="T"))
    creator.llm_client.config.generate.assert_not_called()


def test_creator_without_book_text_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIFY_METADATA_LOOKUP", "1")
    creator = object.__new__(AudiobookCreator)
    creator.reader = object()  # no .book
    creator.llm_client = MagicMock()
    creator.library_root = tmp_path
    creator._infer_from_text(BookMetadata(title="T"))
    creator.llm_client.config.generate.assert_not_called()
