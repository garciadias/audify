"""Tests for adopting outputs made before Library items existed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner
from ebooklib import epub

from audify.cli import cli
from audify.library.adopt import adopt, find_old_outputs, index_ebooks, match_ebook
from audify.library.publish import MANIFEST
from audify.utils.m4b_builder import (
    append_chapter_metadata,
    assemble_m4b,
    write_metadata_header,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


def _epub(
    path: Path,
    title: str,
    author: str,
    text: str = "Chapter text.",
    cover: bool = False,
) -> Path:
    book = epub.EpubBook()
    if cover:
        book.set_cover("cover.jpg", b"\xff\xd8\xff\xe0epub-cover")
    book.set_identifier(title)
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)
    chapter = epub.EpubHtml(title="One", file_name="one.xhtml", lang="en")
    chapter.content = f"<h1>One</h1><p>{text}</p>"
    book.add_item(chapter)
    book.spine = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book)
    return path


def _m4b(path: Path, chapters: list[tuple[str, int]]) -> Path:
    """An untagged M4B with chapters, as old runs produced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".raw.m4a")
    seconds = sum(d for _, d in chapters)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=f=440:d={seconds}",
            "-c:a",
            "aac",
            "-f",
            "mp4",
            str(raw),
        ],
        check=True,
    )
    md = path.with_suffix(".txt")
    write_metadata_header(md, {})
    start_ms = 0
    for title, d in chapters:
        start_ms = append_chapter_metadata(md, title, start_ms, d)
    assemble_m4b(raw, md, path)
    raw.unlink(missing_ok=True)
    md.unlink(missing_ok=True)
    return path


@pytest.fixture
def old_library(tmp_path):
    out = tmp_path / "output"
    books = tmp_path / "books"
    _epub(books / "Book One.epub", "Book One", "Ann Author")
    _epub(books / "Covered.epub", "Covered", "Cy Cover", "covered text", cover=True)
    _m4b(out / "Covered.m4b", [("Only", 2)])
    work = out / "book_one"
    _m4b(work / "book_one.m4b", [("Opening", 2), ("Ending", 2)])
    (work / "cover.jpg").write_bytes(b"\xff\xd8\xffcover")
    (work / "scripts").mkdir()
    (work / "scripts" / "episode_001.txt").write_text("script")
    _m4b(out / "Loose Talk.m4b", [("Only", 2)])
    _m4b(out / "long_book_part1" / "long_book_part1.m4b", [("A", 2)])
    _m4b(out / "long_book_part2" / "long_book_part2.m4b", [("B", 2)])
    return out, books, tmp_path / "library"


def _tags(m4b: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_chapters",
            str(m4b),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_find_old_outputs_groups_parts_and_loose_files(tmp_path):
    out = tmp_path / "out"
    for rel in [
        "a/a.m4b",
        "b_part1/b_part1.m4b",
        "b_part2/b_part2.m4b",
        "Loose.m4b",
        "c/c.tmp.m4b",
        ".hidden/h.m4b",
    ]:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(b"x")
    (out / "done").mkdir()
    (out / "done" / "done.m4b").write_bytes(b"x")
    (out / "done" / MANIFEST).write_text("{}")
    found = {o.name: [p.name for p in o.audio] for o in find_old_outputs(out)}
    assert found == {
        "a": ["a.m4b"],
        "b": ["b_part1.m4b", "b_part2.m4b"],
        "loose": ["Loose.m4b"],
    }


def test_ebook_matching(tmp_path):
    _epub(tmp_path / "x" / "Book One.epub", "Book One", "A", "same")
    _epub(tmp_path / "y" / "Book One.epub", "Book One (fixed)", "A", "same")
    _epub(tmp_path / "x" / "Twin.epub", "Twin", "A", "first")
    _epub(tmp_path / "y" / "Twin.epub", "Twin", "A", "second")
    index = index_ebooks([tmp_path / "x", tmp_path / "y", tmp_path / "missing"])
    ebook, note = match_ebook("book_one", index)
    assert ebook is not None
    assert note == ""  # two copies, one Book key: fine
    assert match_ebook("twin", index)[1].startswith("unmatched (ambiguous")
    assert match_ebook("nothing", index) == (None, "unmatched")


@needs_ffmpeg
def test_dry_run_lists_the_plan_and_changes_nothing(old_library):
    out, books, lib = old_library
    before = sorted(p for p in out.rglob("*"))
    lines: list[str] = []
    plans = adopt(out, lib, [books], echo=lines.append)
    text = "\n".join(lines)

    assert sorted(p for p in out.rglob("*")) == before
    assert not lib.exists()
    assert len(plans) == 4
    assert "Book One.epub" in text
    assert "target: Ann Author/Book One/" in text
    assert "unmatched" in text
    assert "Long Book - Part 01.m4b" in text
    assert "Dry run" in text


@needs_ffmpeg
def test_apply_publishes_items_and_keeps_work_files(old_library):
    out, books, lib = old_library
    adopt(
        out,
        lib,
        [books],
        apply=True,
        narrators=["Kokoro (af_bella)"],
        echo=lambda s: None,
    )

    item = lib / "Ann Author" / "Book One"
    assert sorted(p.name for p in item.iterdir()) == [
        ".audify.json",
        "Book One.epub",
        "Book One.m4b",
        "cover.jpg",
        "metadata.json",
    ]
    probe = _tags(item / "Book One.m4b")
    assert probe["format"]["tags"]["title"] == "Book One"
    assert probe["format"]["tags"]["artist"] == "Ann Author"
    assert [c["tags"]["title"] for c in probe["chapters"]] == ["Opening", "Ending"]
    data = json.loads((item / "metadata.json").read_text())
    assert data["narrators"] == ["Kokoro (af_bella)"]
    assert [c["title"] for c in data["chapters"]] == ["Opening", "Ending"]

    # The old M4B and cover are gone; the work folder and scripts remain.
    assert not (out / "book_one" / "book_one.m4b").exists()
    assert not (out / "book_one" / "cover.jpg").exists()
    assert (out / "book_one" / "scripts" / "episode_001.txt").exists()

    # Unmatched: published without an ebook, titled from the file name.
    loose = lib / "Unknown Author" / "Loose Talk"
    assert not list(loose.glob("*.epub"))
    assert (loose / "Loose Talk.m4b").exists()

    # A loose M4B with no work folder takes its EPUB's cover.
    covered = lib / "Cy Cover" / "Covered"
    assert (covered / "cover.jpg").read_bytes() == b"\xff\xd8\xff\xe0epub-cover"
    assert (
        json.loads((covered / MANIFEST).read_text())["provenance"]["cover"] == "ebook"
    )

    # Parts: one item, track n/N, one continuous chapter list.
    long = lib / "Unknown Author" / "Long Book"
    assert _tags(long / "Long Book - Part 02.m4b")["format"]["tags"]["track"] == "2/2"
    chapters = json.loads((long / "metadata.json").read_text())["chapters"]
    assert [c["title"] for c in chapters] == ["A", "B"]
    assert chapters[1]["start"] == pytest.approx(chapters[0]["end"], abs=0.1)
    assert not list(lib.rglob(".*.adopt.m4b"))
    assert not list(out.rglob(".*.adopt.m4b"))


@needs_ffmpeg
def test_adopting_again_is_idempotent(old_library):
    out, books, lib = old_library
    adopt(out, lib, [books], apply=True, echo=lambda s: None)
    snapshot = {p: p.read_bytes() for p in lib.rglob("*") if p.is_file()}
    lines: list[str] = []
    plans = adopt(out, lib, [books], apply=True, echo=lines.append)
    assert plans == []
    assert {p: p.read_bytes() for p in lib.rglob("*") if p.is_file()} == snapshot


@needs_ffmpeg
def test_cli_publish_adopt(old_library):
    out, books, lib = old_library
    result = CliRunner().invoke(
        cli,
        [
            "publish",
            "--adopt",
            "--output",
            str(out),
            "--library-root",
            str(lib),
            "--ebooks",
            str(books),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert not lib.exists()


def test_cli_refuses_overlapping_folders(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "publish",
            "--adopt",
            "--output",
            str(tmp_path),
            "--library-root",
            str(tmp_path / "lib"),
        ],
    )
    assert result.exit_code != 0
