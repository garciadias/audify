"""Book key and Item identity: how a book is recognised across runs.

The Book key is a hash of the book's text, so a metadata fix made in Calibre
(which rewrites the OPF, and therefore the file) keeps the key, while a
different edition gets a new one (ADR-0004).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import bs4
import ebooklib
from ebooklib import epub

# Bump when the computation changes (normalisation, which items count): every
# existing Item manifest records the version its key was made with.
KEY_VERSION = 1

_WS = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def book_key_from_text(text: str) -> str:
    """Book key of already extracted text (used for PDFs)."""
    digest = hashlib.sha256(_normalise(text).encode("utf-8")).hexdigest()
    return f"v{KEY_VERSION}:{digest}"


def epub_spine_text(book: epub.EpubBook) -> str:
    """Visible text of the EPUB spine, in reading order.

    The OPF and the cover image are not part of it, so editing the metadata
    leaves this text, and the Book key, unchanged. The navigation document is
    skipped because EPUB writers regenerate it.
    """
    parts: list[str] = []
    for entry in book.spine:
        idref = entry[0] if isinstance(entry, (tuple, list)) else entry
        item = book.get_item_with_id(idref)
        if item is None or isinstance(item, epub.EpubNav):
            continue
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = bs4.BeautifulSoup(item.get_content(), "html.parser")
        text = _normalise(soup.get_text(" "))
        if text:
            parts.append(text)
    return "\n".join(parts)


def book_key_from_epub(path: str | Path) -> str:
    """Book key of an EPUB file."""
    return book_key_from_text(epub_spine_text(epub.read_epub(str(path))))


@dataclass(frozen=True)
class ItemIdentity:
    """What Re-publish matches on: the Book key plus the rendition kind."""

    book_key: str
    task: str = "audiobook"
    language: Optional[str] = None  # translation target, if any

    def to_json(self) -> dict:
        return {
            "book_key": self.book_key,
            "key_version": KEY_VERSION,
            "task": self.task,
            "language": self.language,
        }

    @classmethod
    def from_json(cls, data: dict) -> "ItemIdentity":
        return cls(
            book_key=str(data.get("book_key", "")),
            task=str(data.get("task", "audiobook")),
            language=data.get("language"),
        )
