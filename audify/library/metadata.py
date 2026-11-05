"""Book metadata: the single record every metadata carrier is generated from.

Book metadata is built from the Source ebook and written, unchanged, into
every carrier: the tags embedded in the M4B (read by any player) and the
Audiobookshelf ``metadata.json`` sidecar. See CONTEXT.md, "Library delivery".
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ebooklib import epub

logger = logging.getLogger(__name__)

_ISBN_RE = re.compile(r"(97[89][\d-]{10,14}|\d[\d-]{8,11}[\dXx])")
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SeriesEntry:
    """A series the book belongs to, with its position when known."""

    name: str
    sequence: Optional[str] = None

    def label(self) -> str:
        """Audiobookshelf's series notation: ``Name #N`` or ``Name``."""
        return f"{self.name} #{self.sequence}" if self.sequence else self.name


METADATA_FIELDS = (
    "title",
    "subtitle",
    "authors",
    "narrators",
    "series",
    "genres",
    "tags",
    "published_date",
    "publisher",
    "description",
    "isbn",
    "language",
)


@dataclass
class BookMetadata:
    """Identity of one book. Empty fields stay empty, never placeholders."""

    title: str = ""
    subtitle: str = ""
    authors: list[str] = field(default_factory=list)
    narrators: list[str] = field(default_factory=list)
    series: list[SeriesEntry] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    published_date: str = ""
    publisher: str = ""
    description: str = ""
    isbn: str = ""
    language: str = ""
    # Where each non-empty field came from: "override", "ebook", a catalogue's
    # name, or "llm". Recorded in the Item manifest, not in any carrier.
    provenance: dict[str, str] = field(default_factory=dict, compare=False)

    def filled_fields(self) -> list[str]:
        """Names of the metadata fields that currently hold a value."""
        return [name for name in METADATA_FIELDS if getattr(self, name)]

    def mark(self, source: str, names: Optional[Iterable[str]] = None) -> None:
        """Record *source* as the provenance of *names* (default: every filled
        field that has no provenance yet)."""
        for name in names if names is not None else self.filled_fields():
            if getattr(self, name) and name not in self.provenance:
                self.provenance[name] = source

    @property
    def published_year(self) -> str:
        """The four-digit year of ``published_date``, or an empty string."""
        match = re.search(r"\d{4}", self.published_date)
        return match.group(0) if match else ""

    def m4b_tags(
        self, track: Optional[int] = None, total_tracks: Optional[int] = None
    ) -> dict[str, str]:
        """Tags to embed in an M4B, using only standard MP4 (iTunes) keys.

        Keys outside the iTunes atom set (language, publisher, ISBN,
        subtitle) are left to ``metadata.json``. Writing them into the M4B
        needs FFmpeg's ``use_metadata_tags``, which replaces the iTunes
        atoms that other players read.
        """
        tags: dict[str, str] = {}
        if self.title:
            tags["title"] = self.title
            tags["album"] = self.title
        if self.authors:
            joined = ", ".join(self.authors)
            tags["artist"] = joined
            tags["album_artist"] = joined
        if self.narrators:
            tags["composer"] = ", ".join(self.narrators)
        if self.published_year:
            tags["date"] = self.published_year
        if self.genres:
            tags["genre"] = ", ".join(self.genres)
        if self.description:
            tags["comment"] = self.description
            tags["description"] = self.description
        if self.series:
            first = self.series[0]
            tags["show"] = first.name
            tags["grouping"] = first.name
            if first.sequence:
                tags["episode_id"] = first.sequence
        if track is not None:
            tags["track"] = f"{track}/{total_tracks}" if total_tracks else str(track)
        return tags


def _first(values: Iterable[tuple[Any, dict]]) -> str:
    for value, _attrs in values:
        if value and str(value).strip():
            return _clean(str(value))
    return ""


def _clean(text: str) -> str:
    """Strip markup and collapse whitespace (OPF descriptions are often HTML)."""
    text = html.unescape(_TAG_RE.sub(" ", text))
    return _WS_RE.sub(" ", text).strip()


OPF_NS = "http://www.idpf.org/2007/opf"


def _authors(book: epub.EpubBook) -> list[str]:
    """dc:creator entries that are authors, split and tidied.

    Creators with an OPF role other than ``aut`` (translator, editor,
    illustrator) are not authors. Some tools write several names into one
    creator separated by ";", or leave a trailing separator.
    """
    # EPUB 3 states roles in <meta refines="#id" property="role">; EPUB 2 in
    # an opf:role attribute on the creator itself.
    refined: dict[str, str] = {}
    all_meta = getattr(book, "metadata", None)
    for namespace in (OPF_NS, None):
        section = all_meta.get(namespace) if isinstance(all_meta, dict) else None
        entries = section.get("meta", []) if isinstance(section, dict) else []
        for value, attrs in entries:
            attrs = attrs if isinstance(attrs, dict) else {}
            if attrs.get("property") == "role" and attrs.get("refines") and value:
                refined[str(attrs["refines"]).lstrip("#")] = str(value)
    authors: list[str] = []
    for value, attrs in _dc(book, "creator"):
        if not value:
            continue
        attrs = attrs or {}
        role = refined.get(str(attrs.get("id", "")), "") or next(
            (str(v) for k, v in attrs.items() if str(k).endswith("role")), ""
        )
        if role and role.lower() != "aut":
            continue
        for name in _clean(str(value)).split(";"):
            name = name.strip(" ,;")
            if name and name not in authors:
                authors.append(name)
    return authors


def _dc(book: epub.EpubBook, name: str) -> list[tuple[Any, dict]]:
    try:
        return list(book.get_metadata("DC", name) or [])
    except (KeyError, AttributeError, TypeError):
        return []


def _opf_meta(book: epub.EpubBook) -> list[tuple[Any, dict]]:
    try:
        return list(book.get_metadata("OPF", "meta") or [])
    except (KeyError, AttributeError, TypeError):
        return []


def _isbn(book: epub.EpubBook) -> str:
    for value, attrs in _dc(book, "identifier"):
        raw = str(value or "")
        scheme = " ".join(str(v) for v in (attrs or {}).values()).lower()
        if "isbn" not in scheme and "isbn" not in raw.lower():
            if not re.fullmatch(r"(urn:)?[\d-]{10,17}[\dXx]?", raw.strip()):
                continue
        match = _ISBN_RE.search(raw)
        if match:
            digits = re.sub(r"[^\dXx]", "", match.group(0)).upper()
            if len(digits) in (10, 13):
                return digits
    return ""


def _series(book: epub.EpubBook) -> list[SeriesEntry]:
    """Calibre ``calibre:series`` metas, or EPUB 3 ``belongs-to-collection``."""
    metas = _opf_meta(book)
    by_name = {
        str((attrs or {}).get("name", "")): str((attrs or {}).get("content", ""))
        for _value, attrs in metas
        if attrs and attrs.get("name")
    }
    name = by_name.get("calibre:series", "").strip()
    if name:
        index = by_name.get("calibre:series_index", "").strip()
        if index.endswith(".0"):
            index = index[:-2]
        return [SeriesEntry(name=name, sequence=index or None)]

    collections: dict[str, SeriesEntry] = {}
    positions: dict[str, str] = {}
    for value, attrs in metas:
        attrs = attrs or {}
        prop = attrs.get("property", "")
        if prop == "belongs-to-collection" and value:
            collections[attrs.get("id", str(value))] = SeriesEntry(_clean(str(value)))
        elif prop == "group-position" and value:
            positions[str(attrs.get("refines", "")).lstrip("#")] = str(value).strip()
    return [
        SeriesEntry(entry.name, positions.get(key))
        for key, entry in collections.items()
    ]


def book_metadata_from_epub(
    book: epub.EpubBook, narrators: Optional[list[str]] = None
) -> BookMetadata:
    """Build Book metadata from an EPUB's OPF.

    Args:
        book: The parsed EPUB.
        narrators: The Narrator, as ``Engine (voice)``. There is never a human
            narrator, so the caller supplies it from the TTS settings.
    """
    metadata = BookMetadata(
        title=_first(_dc(book, "title")),
        authors=_authors(book),
        narrators=list(narrators or []),
        series=_series(book),
        genres=[_clean(str(v)) for v, _ in _dc(book, "subject") if v],
        published_date=_first(_dc(book, "date")),
        publisher=_first(_dc(book, "publisher")),
        description=_first(_dc(book, "description")),
        isbn=_isbn(book),
        language=_first(_dc(book, "language")),
    )
    metadata.mark("ebook", [f for f in metadata.filled_fields() if f != "narrators"])
    metadata.mark("audify", ["narrators"])
    return metadata


def narrator_label(engine: str, voice: Optional[str]) -> str:
    """The Narrator, ``Engine (voice)``, for example ``Kokoro (af_bella)``."""
    pretty = engine.strip().capitalize() if engine else "TTS"
    return f"{pretty} ({voice})" if voice else pretty


def title_from_filename(path: Path) -> str:
    """A readable title from a file name: ``the_wandering-earth`` becomes
    ``The Wandering Earth``. Used only when the source has no title."""
    words = re.sub(r"[_\-.]+", " ", Path(path).stem).split()
    # Lowercase words are capitalised; "NLP" and "iPhone" keep their casing.
    return " ".join(w if w != w.lower() else w.capitalize() for w in words)


_PDF_PLACEHOLDERS = {"untitled", "untitled document", "anonymous", "unknown", "user"}


def book_metadata_from_pdf(
    path: Path, narrators: Optional[list[str]] = None
) -> BookMetadata:
    """Book metadata from a PDF's info dictionary (Title, Author, Subject,
    Keywords). An empty dictionary yields a file-name title; enrichment fills
    the rest."""
    import pypdf

    info: dict[str, str] = {}
    try:
        with open(path, "rb") as f:
            raw: Any = pypdf.PdfReader(f).metadata or {}
            # Resolve while the file is open: an indirect value is read from
            # the stream on first use. A PDF null means the field is absent.
            for key in ("/Title", "/Author", "/Subject", "/Keywords"):
                value = raw.get(key)
                if isinstance(value, pypdf.generic.IndirectObject):
                    value = value.get_object()
                if value is not None and not isinstance(
                    value, pypdf.generic.NullObject
                ):
                    info[key] = str(value).strip()
    except Exception as e:
        logger.warning(f"Could not read PDF metadata from {path}: {e}")

    def text(key: str) -> str:
        value = info.get(key, "")
        # PDF generators fill unset fields with placeholders.
        return "" if value.casefold() in _PDF_PLACEHOLDERS else value

    authors = [
        a.strip()
        for a in re.split(r";|\band\b|&", text("/Author"), flags=re.IGNORECASE)
        if a.strip()
    ]
    keywords = [k.strip() for k in re.split(r"[;,]", text("/Keywords")) if k.strip()]
    title = text("/Title")
    # Word processors write the document file name as the title.
    if re.match(r"(?i)^microsoft word\s*-|.*\.(docx?|pdf|tex|odt)$", title):
        title = ""
    metadata = BookMetadata(
        title=title,
        authors=authors,
        description=text("/Subject"),
        tags=keywords,
        narrators=list(narrators or []),
    )
    metadata.mark("ebook", [f for f in metadata.filled_fields() if f != "narrators"])
    metadata.mark("audify", ["narrators"])
    if not metadata.title:
        metadata.title = title_from_filename(path)
        metadata.mark("filename", ["title"])
    return metadata
