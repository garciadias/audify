"""Tests for catalogue enrichment, on recorded Open Library responses."""

import json
from pathlib import Path
from typing import Optional

from audify.library.catalogue import (
    GoogleBooks,
    OpenLibrary,
    enrich_from_catalogues,
    fetch_cover,
)
from audify.library.metadata import BookMetadata
from audify.library.override import apply_override

FIX = Path(__file__).parent / "fixtures" / "catalogue"
ISBN = "9780618173877"


def recorded(url: str) -> Optional[bytes]:
    if f"/isbn/{ISBN}.json" in url:
        return (FIX / f"ol_isbn_{ISBN}.json").read_bytes()
    if "/works/" in url:
        return (FIX / "ol_work.json").read_bytes()
    if "/authors/" in url:
        return (FIX / "ol_author.json").read_bytes()
    if "/search.json" in url:
        return (FIX / "ol_search.json").read_bytes()
    if "covers.openlibrary.org" in url:
        return b"\xff\xd8\xff\xe0fakejpeg"
    return None


def test_isbn_lookup_fills_gaps_with_provenance():
    meta = BookMetadata(title="Everything Is Illuminated", isbn=ISBN)
    meta.mark("ebook")
    enrich_from_catalogues(meta, [OpenLibrary(recorded)])
    assert meta.authors == ["Jonathan Safran Foer"]
    assert meta.publisher == "Houghton Mifflin Co."
    assert meta.published_date == "2002"
    assert meta.description.startswith("A young man arrives in the Ukraine")
    assert "Fiction" in meta.genres
    assert not any("--" in g for g in meta.genres)
    assert meta.provenance["authors"] == "openlibrary"
    assert meta.provenance["isbn"] == "ebook"


def test_existing_values_are_never_replaced():
    meta = BookMetadata(title="T", isbn=ISBN, publisher="Penguin", authors=["Me"])
    meta.mark("ebook")
    enrich_from_catalogues(meta, [OpenLibrary(recorded)])
    assert meta.publisher == "Penguin"
    assert meta.authors == ["Me"]


def test_override_cleared_field_stays_empty():
    meta = apply_override(BookMetadata(title="T", isbn=ISBN), {"publisher": None})
    enrich_from_catalogues(meta, [OpenLibrary(recorded)])
    assert meta.publisher == ""


def test_title_search_fills_work_facts_only():
    meta = BookMetadata(title="Everything is illuminated", authors=["J. S. Foer"])
    enrich_from_catalogues(meta, [OpenLibrary(recorded)])
    assert meta.published_date == "2002"
    assert meta.description
    assert meta.isbn == ""  # another edition's ISBN is not this book's
    assert meta.publisher == ""


def test_title_search_rejects_wrong_author():
    meta = BookMetadata(title="Everything Is Illuminated", authors=["Someone Else"])
    assert enrich_from_catalogues(meta, [OpenLibrary(recorded)]) == []
    assert meta.description == ""


def test_offline_leaves_fields_empty():
    meta = BookMetadata(title="T", isbn=ISBN)
    assert enrich_from_catalogues(meta, [OpenLibrary(lambda url: None)]) == []
    assert meta.authors == []


def test_google_books_used_after_open_library():
    body = json.dumps(
        {"items": [{"volumeInfo": {"title": "T", "publisher": "GB Press"}}]}
    ).encode()
    meta = BookMetadata(title="T", isbn="123")
    enrich_from_catalogues(
        meta, [OpenLibrary(lambda u: None), GoogleBooks("k", lambda u: body)]
    )
    assert meta.publisher == "GB Press"
    assert meta.provenance["publisher"] == "googlebooks"


def test_cover_only_saved_when_it_is_a_jpeg(tmp_path):
    meta = BookMetadata(title="T", isbn=ISBN)
    records = enrich_from_catalogues(meta, [OpenLibrary(recorded)])
    assert fetch_cover(records, tmp_path / "c.jpg", recorded) == tmp_path / "c.jpg"
    assert fetch_cover(records, tmp_path / "d.jpg", lambda u: b"<html>") is None
