"""Publish a Library item: one self-contained folder per book.

A Library item is assembled under a hidden (dot-prefixed) name inside the
Library root, then renamed into place, so the audiobook server never scans a
half-built book (ADR-0003). See CONTEXT.md, "Library delivery".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from audify.library.identity import ItemIdentity
from audify.library.metadata import BookMetadata

logger = logging.getLogger(__name__)

LIBRARY_ROOT_ENV = "AUDIFY_LIBRARY_ROOT"
UNKNOWN_AUTHOR = "Unknown Author"
METADATA_JSON = "metadata.json"
COVER_NAME = "cover.jpg"
MANIFEST = ".audify.json"
MANIFEST_VERSION = 1
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _planned_files(
    metadata: BookMetadata,
    audio_files: Sequence[Path],
    source_ebook: Optional[Path],
    cover: Optional[Path],
    chapters: Optional[Sequence[ChapterMark]],
) -> dict[str, Path | bytes]:
    """Every file Audify writes into a Library item: name to source or content."""
    planned: dict[str, Path | bytes] = dict(
        zip(audio_names(metadata, audio_files), audio_files)
    )
    if source_ebook is not None:
        planned[source_ebook.name] = source_ebook
    if cover is not None and cover.exists():
        planned[COVER_NAME] = cover
    planned[METADATA_JSON] = (
        json.dumps(abs_metadata(metadata, chapters), indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    return planned


def _write(dest: Path, source: Path | bytes) -> None:
    """Write *dest* atomically: a hidden temporary sibling, then a rename."""
    tmp = dest.with_name(f".{dest.name}.audify-tmp")
    try:
        if isinstance(source, bytes):
            tmp.write_bytes(source)
        else:
            shutil.copy2(source, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def read_manifest(item_dir: Path) -> Optional[dict[str, Any]]:
    """The Item manifest of *item_dir*, or None when it has none."""
    path = item_dir / MANIFEST
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Unreadable item manifest {path}: {e}")
        return None
    return data if isinstance(data, dict) else None


def _write_manifest(
    item_dir: Path,
    folder: str,
    identity: Optional[ItemIdentity],
    files: dict[str, str],
    ebook: Optional[str],
    extra: Optional[dict[str, Any]] = None,
) -> None:
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "identity": identity.to_json() if identity else None,
        "folder": folder,
        "ebook": ebook,
        "files": dict(sorted(files.items())),
    }
    if extra:
        manifest.update(extra)
    _write(
        item_dir / MANIFEST,
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def find_library_item(library_root: Path, identity: ItemIdentity) -> Optional[Path]:
    """The existing Library item with *identity*, found through its manifest."""
    if not library_root.is_dir():
        return None
    for manifest_path in library_root.rglob(MANIFEST):
        item_dir = manifest_path.parent
        if any(
            part.startswith(".") for part in item_dir.relative_to(library_root).parts
        ):
            continue  # a staging folder, or hidden by the user
        manifest = read_manifest(item_dir)
        if not manifest or not manifest.get("identity"):
            continue
        if ItemIdentity.from_json(manifest["identity"]) == identity:
            return item_dir
    return None


def _republish(
    library_root: Path,
    item_dir: Path,
    planned: dict[str, Path | bytes],
    identity: Optional[ItemIdentity],
    ebook: Optional[str],
    extra: Optional[dict[str, Any]],
) -> Path:
    """Update an existing Library item in place, keeping its folder path.

    A file whose hash differs from what Audify last wrote was edited by hand:
    it is kept and dropped from the manifest, so it stays the user's from then
    on. Files Audify never wrote are untouched. The Source ebook is never
    replaced or deleted.
    """
    manifest = read_manifest(item_dir) or {}
    recorded: dict[str, str] = dict(manifest.get("files") or {})
    old_ebook = manifest.get("ebook")
    written: dict[str, str] = {}

    for name, source in planned.items():
        dest = item_dir / name
        if dest.exists():
            if name not in recorded:
                logger.info(f"Keeping {dest.name}: not written by Audify.")
                continue
            if _sha256(dest) != recorded[name]:
                logger.warning(
                    f"Keeping {dest.name}: edited by hand since last publish."
                )
                continue
            if name in (ebook, old_ebook):
                written[name] = recorded[name]
                continue
        _write(dest, source)
        written[name] = _sha256(dest)

    for name, digest in recorded.items():
        if name in planned or name in (ebook, old_ebook):
            continue
        stale = item_dir / name
        if stale.exists() and _sha256(stale) == digest:
            stale.unlink()
            logger.info(f"Removed {name}: no longer part of this book.")

    folder = manifest.get("folder") or item_dir.relative_to(library_root).as_posix()
    _write_manifest(item_dir, folder, identity, written, ebook or old_ebook, extra)
    logger.info(f"Re-published Library item: {item_dir}")
    return item_dir


def publish_library_item(
    library_root: Path,
    metadata: BookMetadata,
    audio_files: Sequence[Path],
    source_ebook: Optional[Path] = None,
    cover: Optional[Path] = None,
    chapters: Optional[Sequence[ChapterMark]] = None,
    identity: Optional[ItemIdentity] = None,
    manifest_extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Publish a Library item under *library_root*, or Re-publish an existing one.

    Args:
        library_root: The folder the audiobook server scans.
        metadata: Book metadata; it names the folder and fills metadata.json.
        audio_files: The M4B, or every Part in order.
        source_ebook: The EPUB or PDF the run was made from, copied in as the
            server's ebook. None for a Derived work.
        cover: Cover image, copied as ``cover.jpg``.
        chapters: The whole-book chapter list for metadata.json.
        identity: The Item identity. When an item with this identity already
            exists, it is updated in place and keeps its folder path.
        manifest_extra: Extra keys for the Item manifest (provenance, ...).

    Returns:
        The Library item folder.

    Raises:
        FileExistsError: When the target folder exists but is not this book's
            Library item (no matching manifest).
        ValueError: When there is no audio to publish.
    """
    if not audio_files:
        raise ValueError("No audio files to publish.")
    library_root.mkdir(parents=True, exist_ok=True)
    planned = _planned_files(metadata, audio_files, source_ebook, cover, chapters)
    ebook = source_ebook.name if source_ebook is not None else None

    if identity is not None:
        existing = find_library_item(library_root, identity)
        if existing is not None:
            return _republish(
                library_root, existing, planned, identity, ebook, manifest_extra
            )

    final_dir = library_root / item_relative_dir(metadata)
    if final_dir.exists():
        raise FileExistsError(f"Library item already exists: {final_dir}")

    staging = library_root / f"{_TMP_PREFIX}{uuid.uuid4().hex[:12]}"
    staging.mkdir()
    created_parents = [p for p in final_dir.parents if not p.exists()]
    try:
        written: dict[str, str] = {}
        for name, source in planned.items():
            _write(staging / name, source)
            written[name] = _sha256(staging / name)
        # The manifest goes in before the rename, so the item never appears
        # in the library without it.
        folder = final_dir.relative_to(library_root).as_posix()
        _write_manifest(staging, folder, identity, written, ebook, manifest_extra)
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
