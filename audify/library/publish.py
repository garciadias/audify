"""Publish a Library item: one self-contained folder per book.

A Library item is assembled under a hidden (dot-prefixed) name inside the
Library root, then renamed into place, so the audiobook server never scans a
half-built book (ADR-0003). See CONTEXT.md, "Library delivery".
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from audify.library.metadata import BookMetadata

logger = logging.getLogger(__name__)

LIBRARY_ROOT_ENV = "AUDIFY_LIBRARY_ROOT"
UNKNOWN_AUTHOR = "Unknown Author"
METADATA_JSON = "metadata.json"
COVER_NAME = "cover.jpg"
_TMP_PREFIX = ".audify-publishing-"

# Characters that are invalid on Windows or FAT/exFAT SD cards (the library is
# synced to an Android phone), plus control characters.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_NAME_BYTES = 120


@dataclass(frozen=True)
class ChapterMark:
    """One chapter of the whole book, in seconds from the start."""

    title: str
    start: float
    end: float


def safe_name(name: str) -> str:
    """A folder or file name that is valid on Linux, Android and FAT media."""
    cleaned = _UNSAFE.sub(" ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    encoded = cleaned.encode("utf-8")
    if len(encoded) > _MAX_NAME_BYTES:
        cleaned = encoded[:_MAX_NAME_BYTES].decode("utf-8", "ignore").strip(" .")
    return cleaned


def resolve_library_root(explicit: Optional[str | Path]) -> Optional[Path]:
    """The configured Library root: the explicit value, else the environment."""
    value = explicit or os.environ.get(LIBRARY_ROOT_ENV)
    return Path(value).expanduser().resolve() if value else None


def check_separate(library_root: Path, work_dir: Path) -> None:
    """Refuse a Library root that is, contains, or sits inside the Work directory.

    Raises:
        ValueError: When the two directories overlap (ADR-0003).
    """
    root = library_root.resolve()
    work = work_dir.resolve()
    if root == work or root in work.parents or work in root.parents:
        raise ValueError(
            f"Library root {root} overlaps the work directory {work}; "
            "they must be separate folders (see ADR-0003)."
        )


def item_relative_dir(metadata: BookMetadata) -> Path:
    """``Author/[Series/]Title`` for a Library item."""
    author = safe_name(metadata.authors[0]) if metadata.authors else ""
    parts = [author or UNKNOWN_AUTHOR]
    if metadata.series:
        series = safe_name(metadata.series[0].name)
        if series:
            parts.append(series)
    title = safe_name(metadata.title) or "Untitled"
    parts.append(title)
    return Path(*parts)


def audio_names(metadata: BookMetadata, audio_files: Sequence[Path]) -> list[str]:
    """File names for the audio inside a Library item.

    One file is ``Title.m4b``; Parts are ``Title - Part 01.m4b`` and so on, so
    the order is right even for a player that ignores track tags.
    """
    title = safe_name(metadata.title) or "Untitled"
    if len(audio_files) == 1:
        return [f"{title}{audio_files[0].suffix}"]
    width = max(2, len(str(len(audio_files))))
    return [
        f"{title} - Part {i:0{width}d}{audio.suffix}"
        for i, audio in enumerate(audio_files, start=1)
    ]


def abs_metadata(
    metadata: BookMetadata, chapters: Optional[Sequence[ChapterMark]] = None
) -> dict[str, Any]:
    """Audiobookshelf ``metadata.json`` content for *metadata*.

    Only keys Audiobookshelf accepts are written. ``chapters`` is included only
    when given, because an empty list would erase the chapters read from the
    audio files.
    """
    data: dict[str, Any] = {
        "tags": list(metadata.tags),
        "title": metadata.title or None,
        "subtitle": metadata.subtitle or None,
        "authors": list(metadata.authors),
        "narrators": list(metadata.narrators),
        "series": [entry.label() for entry in metadata.series],
        "genres": list(metadata.genres),
        "publishedYear": metadata.published_year or None,
        "publishedDate": metadata.published_date or None,
        "publisher": metadata.publisher or None,
        "description": metadata.description or None,
        "isbn": metadata.isbn or None,
        "asin": None,
        "language": metadata.language or None,
        "explicit": False,
        "abridged": False,
    }
    if chapters:
        data["chapters"] = [
            {
                "id": i,
                "start": round(c.start, 3),
                "end": round(c.end, 3),
                "title": c.title,
            }
            for i, c in enumerate(chapters)
        ]
    return data


def publish_library_item(
    library_root: Path,
    metadata: BookMetadata,
    audio_files: Sequence[Path],
    source_ebook: Optional[Path] = None,
    cover: Optional[Path] = None,
    chapters: Optional[Sequence[ChapterMark]] = None,
) -> Path:
    """Assemble a Library item and move it into place under *library_root*.

    Args:
        library_root: The folder the audiobook server scans.
        metadata: Book metadata; it names the folder and fills metadata.json.
        audio_files: The M4B, or every Part in order.
        source_ebook: The EPUB or PDF the run was made from, copied in as the
            server's ebook. None for a Derived work.
        cover: Cover image, copied as ``cover.jpg``.
        chapters: The whole-book chapter list for metadata.json.

    Returns:
        The final Library item folder.

    Raises:
        FileExistsError: When the item folder already exists (Re-publish is
            handled separately).
        ValueError: When there is no audio to publish.
    """
    if not audio_files:
        raise ValueError("No audio files to publish.")
    library_root.mkdir(parents=True, exist_ok=True)
    final_dir = library_root / item_relative_dir(metadata)
    if final_dir.exists():
        raise FileExistsError(f"Library item already exists: {final_dir}")

    staging = library_root / f"{_TMP_PREFIX}{uuid.uuid4().hex[:12]}"
    staging.mkdir()
    created_parents = [p for p in final_dir.parents if not p.exists()]
    try:
        for name, audio in zip(audio_names(metadata, audio_files), audio_files):
            shutil.copy2(audio, staging / name)
        if source_ebook is not None:
            shutil.copy2(source_ebook, staging / source_ebook.name)
        if cover is not None and cover.exists():
            shutil.copy2(cover, staging / COVER_NAME)
        (staging / METADATA_JSON).write_text(
            json.dumps(abs_metadata(metadata, chapters), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        # Remove Author/Series folders made for this item, deepest first, so a
        # failed publish leaves no empty folder for the server to list.
        for parent in sorted(created_parents, key=lambda p: len(p.parts), reverse=True):
            try:
                parent.rmdir()
            except OSError:
                pass
        raise
    logger.info(f"Published Library item: {final_dir}")
    return final_dir
