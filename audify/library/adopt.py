"""Adopt: turn outputs made before Library items existed into Library items.

Old runs left ``<output>/<name>/<name>.m4b`` (or a loose ``<output>/X.m4b``)
next to the run's working files, with no book tags. Adopting one pairs it
with its Source ebook, builds Book metadata (override, ebook, catalogues),
writes the tags, ``metadata.json`` and the Item manifest, and publishes a
Library item, all without new TTS.

Pairing: the old output folder is named after the ebook's file name
(``get_file_name_title``), so an ebook whose cleaned file name equals the
output's name is a candidate; candidates with different Book keys are
ambiguous and none is used. ``_partN`` outputs of one book become its Parts.

After the Library item exists, only the old M4B (and a loose ``cover.jpg``)
is removed: the old folder is the book's work directory and keeps its
scripts, episodes and reports. Adopting is a dry run unless ``apply`` is set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ebooklib import epub

from audify.library.catalogue import enrich_from_catalogues, fetch_cover, lookup_enabled
from audify.library.identity import ItemIdentity, book_key_from_epub, book_key_from_text
from audify.library.metadata import (
    BookMetadata,
    book_metadata_from_epub,
    book_metadata_from_pdf,
    title_from_filename,
)
from audify.library.override import OVERRIDE_FILE, apply_override, load_override
from audify.library.publish import (
    MANIFEST,
    ChapterMark,
    audio_names,
    find_library_item,
    item_relative_dir,
    publish_library_item,
)
from audify.utils.m4b_builder import run_ffmpeg
from audify.utils.text import get_file_name_title

logger = logging.getLogger(__name__)

EBOOK_SUFFIXES = (".epub", ".pdf")
_PART_RE = re.compile(r"^(?P<base>.+?)_part(?P<n>\d+)$")


@dataclass
class OldOutput:
    """One book's old output: its name, its M4B(s) and its work folder."""

    name: str
    audio: list[Path]
    work_dir: Optional[Path]  # None for a loose M4B at the top of the output


@dataclass
class AdoptPlan:
    output: OldOutput
    metadata: BookMetadata
    ebook: Optional[Path]
    identity: ItemIdentity
    target: Path  # relative to the Library root
    files: list[str]
    note: str = ""
    cover: Optional[Path] = None
    removals: list[Path] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_and_part(name: str) -> tuple[str, int]:
    match = _PART_RE.match(name)
    return (match["base"], int(match["n"])) if match else (name, 0)


def find_old_outputs(output_dir: Path) -> list[OldOutput]:
    """Old outputs under *output_dir*, ``_partN`` siblings grouped as Parts."""
    found: dict[str, list[tuple[int, Path, Optional[Path]]]] = {}
    for entry in sorted(output_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if (
            entry.is_file()
            and entry.suffix == ".m4b"
            and not entry.name.endswith(".tmp.m4b")
        ):
            base, part = _base_and_part(get_file_name_title(entry.stem))
            found.setdefault(base, []).append((part, entry, None))
        elif entry.is_dir() and not (entry / MANIFEST).exists():
            for m4b in sorted(entry.glob("*.m4b")):
                if m4b.name.endswith(".tmp.m4b"):
                    continue
                base, part = _base_and_part(entry.name)
                if part == 0:
                    _, part = _base_and_part(m4b.stem)
                found.setdefault(base, []).append((part, m4b, entry))
    outputs = []
    for name, items in sorted(found.items()):
        items.sort(key=lambda t: (t[0], t[1].name))
        work_dirs = [w for _, _, w in items if w is not None]
        outputs.append(
            OldOutput(
                name=name,
                audio=[p for _, p, _ in items],
                work_dir=work_dirs[0] if work_dirs else None,
            )
        )
    return outputs


def index_ebooks(dirs: Sequence[Path]) -> dict[str, list[Path]]:
    """EPUB and PDF files under *dirs*, by cleaned file name."""
    index: dict[str, list[Path]] = {}
    for directory in dirs:
        if not directory.is_dir():
            logger.warning(f"Ebook folder not found: {directory}")
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in EBOOK_SUFFIXES and path.is_file():
                index.setdefault(get_file_name_title(path.stem), []).append(path)
    return index


def _book_key(ebook: Path) -> str:
    if ebook.suffix.lower() == ".pdf":
        from audify.readers.pdf import PdfReader

        return book_key_from_text(PdfReader(ebook).cleaned_text)
    return book_key_from_epub(ebook)


def match_ebook(name: str, index: dict[str, list[Path]]) -> tuple[Optional[Path], str]:
    """The Source ebook for an old output named *name*, and why if none."""
    candidates = index.get(name, [])
    if not candidates:
        return None, "unmatched"
    keys: dict[str, Path] = {}
    for path in candidates:
        try:
            keys.setdefault(_book_key(path), path)
        except Exception as e:
            logger.warning(f"Cannot read {path}: {e}")
    if len(keys) > 1:
        names = ", ".join(str(p) for p in keys.values())
        return None, f"unmatched (ambiguous: {names})"
    if not keys:
        return None, "unmatched (unreadable ebook)"
    return next(iter(keys.values())), ""


def read_chapters(m4b: Path, offset: float = 0.0) -> list[ChapterMark]:
    """Embedded chapters of *m4b*, shifted by *offset* seconds."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_chapters",
                str(m4b),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        chapters = json.loads(result.stdout).get("chapters", [])
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        json.JSONDecodeError,
    ) as e:
        logger.warning(f"Cannot read chapters of {m4b.name}: {e}")
        return []
    return [
        ChapterMark(
            str(c.get("tags", {}).get("title") or f"Chapter {i + 1}"),
            offset + float(c["start_time"]),
            offset + float(c["end_time"]),
        )
        for i, c in enumerate(chapters)
    ]


def _duration(m4b: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(m4b),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0.0


def retag(m4b: Path, dest: Path, tags: dict[str, str]) -> Path:
    """Copy *m4b* to *dest* with *tags*, keeping audio, chapters and cover."""
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(m4b),
        "-map",
        "0",
        "-map_chapters",
        "0",
        "-c",
        "copy",
    ]
    for key, value in tags.items():
        command += ["-metadata", f"{key}={value}"]
    command += ["-f", "mp4", str(dest)]
    run_ffmpeg(command)
    return dest


def _epub_cover(ebook: Path, dest_dir: Path) -> Optional[Path]:
    from audify.readers.ebook import EpubReader

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        return EpubReader(ebook).get_cover_image(dest_dir)
    except Exception as e:
        logger.warning(f"No cover from {ebook.name}: {e}")
        return None


def plan_adoption(
    library_root: Path,
    output: OldOutput,
    ebook_index: dict[str, list[Path]],
    narrators: Optional[list[str]] = None,
    scratch_dir: Optional[Path] = None,
) -> AdoptPlan:
    scratch_dir = scratch_dir or Path(tempfile.mkdtemp(prefix="audify-adopt-"))
    ebook, note = match_ebook(output.name, ebook_index)
    narrators = list(narrators or [])
    if ebook is None:
        metadata = BookMetadata(
            title=title_from_filename(Path(output.name)), narrators=narrators
        )
        metadata.mark("filename", ["title"])
        metadata.mark("audify", ["narrators"])
        # No ebook, so no Book key: the first audio file identifies the item.
        identity = ItemIdentity(f"audio:{_sha256(output.audio[0])}")
    elif ebook.suffix.lower() == ".pdf":
        metadata = book_metadata_from_pdf(ebook, narrators=narrators)
        identity = ItemIdentity(_book_key(ebook), task="book")
    else:
        metadata = book_metadata_from_epub(
            epub.read_epub(str(ebook)), narrators=narrators
        )
        identity = ItemIdentity(_book_key(ebook), task="book")

    if output.work_dir is not None:
        override = load_override(output.work_dir / OVERRIDE_FILE)
        if override:
            apply_override(metadata, override)
    records = enrich_from_catalogues(metadata) if lookup_enabled() else []

    # Cover: the old run's, else the EPUB's, else a catalogue's.
    cover_dir = output.work_dir or scratch_dir
    cover = None
    if output.work_dir is not None and (output.work_dir / "cover.jpg").exists():
        cover = output.work_dir / "cover.jpg"
    if cover is None and ebook is not None and ebook.suffix.lower() == ".epub":
        cover = _epub_cover(ebook, scratch_dir / output.name)
        if cover is not None:
            metadata.provenance["cover"] = "ebook"
    if cover is None and records:
        cover = fetch_cover(records, cover_dir / f"{output.name}.catalogue.jpg")
        if cover is not None:
            metadata.provenance["cover"] = records[0].source

    existing = find_library_item(library_root, identity)
    target = (
        existing.relative_to(library_root) if existing else item_relative_dir(metadata)
    )
    files = audio_names(metadata, output.audio)
    if ebook is not None:
        files.append(ebook.name)
    if cover is not None:
        files.append("cover.jpg")
    files.append("metadata.json")
    removals = list(output.audio)
    if output.work_dir is not None and (output.work_dir / "cover.jpg").exists():
        removals.append(output.work_dir / "cover.jpg")
    return AdoptPlan(
        output=output,
        metadata=metadata,
        ebook=ebook,
        identity=identity,
        target=target,
        files=files,
        note=note,
        cover=cover,
        removals=removals,
    )


def apply_adoption(library_root: Path, plan: AdoptPlan) -> Path:
    """Publish *plan*'s Library item, then remove the old M4B(s) and cover."""
    total = len(plan.output.audio)
    staged: list[Path] = []
    chapters: list[ChapterMark] = []
    offset = 0.0
    try:
        for n, m4b in enumerate(plan.output.audio, start=1):
            tags = plan.metadata.m4b_tags(
                track=n if total > 1 else None,
                total_tracks=total if total > 1 else None,
            )
            staged.append(retag(m4b, m4b.with_name(f".{m4b.stem}.adopt.m4b"), tags))
            chapters += read_chapters(m4b, offset)
            offset += _duration(m4b)
        item = publish_library_item(
            library_root,
            plan.metadata,
            staged,
            source_ebook=plan.ebook,
            cover=plan.cover,
            chapters=chapters or None,
            identity=plan.identity,
            manifest_extra={
                "provenance": dict(plan.metadata.provenance),
                "adopted_from": [str(p) for p in plan.output.audio],
            },
        )
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    # Only now that the item exists: the old copies go, the work folder stays.
    for path in plan.removals:
        path.unlink(missing_ok=True)
    return item


def format_plan(plan: AdoptPlan) -> str:
    lines = [f"{', '.join(str(p) for p in plan.output.audio)}"]
    lines.append(f"  ebook:  {plan.ebook or plan.note}")
    lines.append(f"  target: {plan.target}/")
    lines += [f"    {name}" for name in plan.files]
    lines += [f"  remove: {path}" for path in plan.removals]
    return "\n".join(lines)


def adopt(
    output_dir: Path,
    library_root: Path,
    ebook_dirs: Sequence[Path],
    apply: bool = False,
    narrators: Optional[list[str]] = None,
    echo: Callable[[str], Any] = print,
) -> list[AdoptPlan]:
    """Plan (and with *apply*, carry out) the adoption of *output_dir*."""
    outputs = find_old_outputs(output_dir)
    index = index_ebooks(ebook_dirs)
    plans: list[AdoptPlan] = []
    scratch = Path(tempfile.mkdtemp(prefix="audify-adopt-"))
    try:
        _adopt_all(library_root, outputs, index, narrators, apply, echo, plans, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    unmatched = [p for p in plans if p.ebook is None]
    echo(
        f"\n{len(plans)} old output(s), {len(unmatched)} without an ebook. "
        + ("Applied." if apply else "Dry run: nothing changed; add --apply.")
    )
    return plans


def _adopt_all(
    library_root: Path,
    outputs: list[OldOutput],
    index: dict[str, list[Path]],
    narrators: Optional[list[str]],
    apply: bool,
    echo: Callable[[str], Any],
    plans: list[AdoptPlan],
    scratch: Path,
) -> None:
    """Plan each output, and apply it before the scratch folder (which may
    hold its extracted cover) is removed."""
    for output in outputs:
        try:
            plan = plan_adoption(library_root, output, index, narrators, scratch)
        except Exception as e:
            echo(f"SKIP {output.name}: {e}")
            continue
        plans.append(plan)
        echo(format_plan(plan))
        if apply:
            try:
                item = apply_adoption(library_root, plan)
                echo(f"  done:   {item}")
            except Exception as e:
                echo(f"  FAILED: {e} (old files kept)")
