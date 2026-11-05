"""Metadata enrichment, rung 1: catalogue lookup.

Empty Book metadata fields are filled from public catalogues: Open Library
first, then Google Books (only when ``GOOGLE_BOOKS_API_KEY`` is set, because
the anonymous quota is shared and routinely exhausted). A book is matched by
ISBN, else by title and author. The policy is fill gaps only: a field that
already holds a value (from an override or the Source ebook) is never
replaced, and every filled field records the catalogue as its provenance.

Lookups never fail a run: an unreachable catalogue, a rate limit or no
confident match leaves the fields empty and logs why.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from audify.library.metadata import BookMetadata

logger = logging.getLogger(__name__)

USER_AGENT = "audify (+https://github.com/garciadias/audify)"
GOOGLE_BOOKS_KEY_ENV = "GOOGLE_BOOKS_API_KEY"
LOOKUP_ENV = "AUDIFY_METADATA_LOOKUP"
_TIMEOUT_S = 15
_MAX_GENRES = 5

# A fetcher returns the body for a URL, or None on any failure (HTTP error,
# rate limit, timeout, no network). Tests inject recorded responses.
Fetcher = Callable[[str], Optional[bytes]]


def http_fetch(url: str) -> Optional[bytes]:
    """GET *url* (following redirects); None on any error."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            logger.warning(f"Catalogue request failed ({e.code}): {url}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning(f"Catalogue unreachable ({e}): {url}")
    return None


def _json(fetch: Fetcher, url: str) -> Optional[dict[str, Any]]:
    body = fetch(url)
    if not body:
        return None
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class CatalogueRecord:
    """What one catalogue knows about a book. Empty fields mean unknown."""

    source: str
    subtitle: str = ""
    authors: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    published_date: str = ""
    publisher: str = ""
    description: str = ""
    isbn: str = ""
    cover_url: str = ""


def _fold(text: str) -> str:
    """Casefolded, accent-free, punctuation-free text for matching titles."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", " ", text.casefold()).strip()


def _same_title(a: str, b: str) -> bool:
    fa = " ".join(_fold(a.split(":")[0]).split())
    fb = " ".join(_fold(b.split(":")[0]).split())
    return bool(fa) and fa == fb


def _author_matches(wanted: list[str], found: list[str]) -> bool:
    if not wanted:
        return True
    surnames = {_fold(name).split()[-1] for name in wanted if _fold(name).split()}
    return any(
        _fold(name).split() and _fold(name).split()[-1] in surnames for name in found
    )


def _clean_genres(subjects: list[Any]) -> list[str]:
    genres: list[str] = []
    for subject in subjects:
        name = subject.get("name") if isinstance(subject, dict) else subject
        if not isinstance(name, str):
            continue
        name = name.strip().rstrip(".")
        # Library-of-Congress strings ("Ukraine -- Fiction") are not genres.
        if not name or "--" in name or len(name) > 40 or name in genres:
            continue
        genres.append(name)
        if len(genres) == _MAX_GENRES:
            break
    return genres


def _text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    return value.strip() if isinstance(value, str) else ""


class OpenLibrary:
    name = "openlibrary"
    base = "https://openlibrary.org"

    def __init__(self, fetch: Fetcher = http_fetch):
        self.fetch = fetch

    def _work(self, work_key: str) -> dict[str, Any]:
        return _json(self.fetch, f"{self.base}{work_key}.json") or {}

    def _author_names(self, refs: list[Any]) -> list[str]:
        names: list[str] = []
        for ref in refs:
            key = ref.get("key") if isinstance(ref, dict) else None
            if isinstance(ref, dict) and isinstance(ref.get("author"), dict):
                key = ref["author"].get("key")
            if not key:
                continue
            author = _json(self.fetch, f"{self.base}{key}.json") or {}
            if author.get("name"):
                names.append(str(author["name"]))
        return names

    def by_isbn(self, isbn: str) -> Optional[CatalogueRecord]:
        edition = _json(self.fetch, f"{self.base}/isbn/{isbn}.json")
        if not edition:
            return None
        works = edition.get("works") or []
        work = self._work(works[0]["key"]) if works and works[0].get("key") else {}
        publishers = edition.get("publishers") or []
        return CatalogueRecord(
            source=self.name,
            subtitle=_text(edition.get("subtitle")),
            authors=self._author_names(
                edition.get("authors") or work.get("authors") or []
            ),
            genres=_clean_genres(work.get("subjects") or edition.get("subjects") or []),
            published_date=_text(edition.get("publish_date")),
            publisher=str(publishers[0]) if publishers else "",
            description=_text(work.get("description") or edition.get("description")),
            isbn=isbn,
            cover_url=f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false",
        )

    def by_title(self, title: str, authors: list[str]) -> Optional[CatalogueRecord]:
        query = {
            "title": title,
            "limit": "5",
            "fields": "key,title,author_name,first_publish_year,subject",
        }
        if authors:
            query["author"] = authors[0]
        data = _json(
            self.fetch, f"{self.base}/search.json?{urllib.parse.urlencode(query)}"
        )
        for doc in (data or {}).get("docs", []):
            if not _same_title(title, str(doc.get("title", ""))):
                continue
            if not _author_matches(authors, list(doc.get("author_name") or [])):
                continue
            work = self._work(doc["key"]) if doc.get("key") else {}
            year = doc.get("first_publish_year")
            # Across editions only work-level facts are safe: no ISBN, no publisher.
            return CatalogueRecord(
                source=self.name,
                authors=list(doc.get("author_name") or []),
                genres=_clean_genres(work.get("subjects") or doc.get("subject") or []),
                published_date=str(year) if year else "",
                description=_text(work.get("description")),
            )
        return None


class GoogleBooks:
    name = "googlebooks"
    base = "https://www.googleapis.com/books/v1/volumes"

    def __init__(self, api_key: str, fetch: Fetcher = http_fetch):
        self.api_key = api_key
        self.fetch = fetch

    def _search(self, q: str) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode(
            {"q": q, "maxResults": "5", "key": self.api_key}
        )
        data = _json(self.fetch, f"{self.base}?{params}") or {}
        return [item.get("volumeInfo") or {} for item in data.get("items") or []]

    def _record(self, info: dict[str, Any], isbn: str = "") -> CatalogueRecord:
        found_isbn = isbn
        if not found_isbn:
            for ident in info.get("industryIdentifiers") or []:
                if ident.get("type") == "ISBN_13":
                    found_isbn = str(ident.get("identifier", ""))
        return CatalogueRecord(
            source=self.name,
            subtitle=_text(info.get("subtitle")),
            authors=list(info.get("authors") or []),
            genres=_clean_genres(info.get("categories") or []),
            published_date=_text(info.get("publishedDate")),
            publisher=_text(info.get("publisher")),
            description=_text(info.get("description")),
            isbn=found_isbn,
        )

    def by_isbn(self, isbn: str) -> Optional[CatalogueRecord]:
        volumes = self._search(f"isbn:{isbn}")
        return self._record(volumes[0], isbn) if volumes else None

    def by_title(self, title: str, authors: list[str]) -> Optional[CatalogueRecord]:
        q = f'intitle:"{title}"' + (f' inauthor:"{authors[0]}"' if authors else "")
        for info in self._search(q):
            if _same_title(title, str(info.get("title", ""))) and _author_matches(
                authors, list(info.get("authors") or [])
            ):
                record = self._record(info)
                record.isbn = ""  # another edition's ISBN is not this book's
                record.publisher = ""
                return record
        return None


# Fields a catalogue may fill. Title and language identify the book and come
# from the ebook or an override; series is not offered by these catalogues.
_FILLABLE = (
    "subtitle",
    "authors",
    "genres",
    "published_date",
    "publisher",
    "description",
    "isbn",
)


def default_catalogues(fetch: Fetcher = http_fetch) -> list[Any]:
    catalogues: list[Any] = [OpenLibrary(fetch)]
    key = os.environ.get(GOOGLE_BOOKS_KEY_ENV)
    if key:
        catalogues.append(GoogleBooks(key, fetch))
    return catalogues


def lookup_enabled() -> bool:
    """Catalogue lookup is on unless AUDIFY_METADATA_LOOKUP is 0/false/off."""
    return os.environ.get(LOOKUP_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def enrich_from_catalogues(
    metadata: BookMetadata, catalogues: Optional[list[Any]] = None
) -> list[CatalogueRecord]:
    """Fill empty fields of *metadata* from *catalogues*, in order.

    Returns the records that matched (for cover lookup and logging).
    """
    catalogues = default_catalogues() if catalogues is None else catalogues
    matched: list[CatalogueRecord] = []
    for catalogue in catalogues:
        if not any(not getattr(metadata, f) for f in _FILLABLE):
            break
        try:
            if metadata.isbn:
                record = catalogue.by_isbn(metadata.isbn)
            elif metadata.title:
                record = catalogue.by_title(metadata.title, metadata.authors)
            else:
                record = None
        except Exception as e:  # a catalogue bug must never fail the run
            logger.warning(f"{catalogue.name} lookup failed: {e}")
            record = None
        if record is None:
            logger.info(f"{catalogue.name}: no confident match for '{metadata.title}'")
            continue
        matched.append(record)
        for name in _FILLABLE:
            value = getattr(record, name)
            # An override that cleared a field means "leave it empty".
            if metadata.provenance.get(name) == "override":
                continue
            if value and not getattr(metadata, name):
                setattr(
                    metadata, name, list(value) if isinstance(value, list) else value
                )
                metadata.provenance[name] = record.source
    return matched


def fetch_cover(
    records: list[CatalogueRecord], dest: Path, fetch: Fetcher = http_fetch
) -> Optional[Path]:
    """Download the first catalogue cover to *dest*; None when there is none."""
    for record in records:
        if not record.cover_url:
            continue
        body = fetch(record.cover_url)
        if body and body[:3] == b"\xff\xd8\xff":  # a real JPEG, not an error page
            dest.write_bytes(body)
            logger.info(f"Cover from {record.source}: {dest.name}")
            return dest
    return None
