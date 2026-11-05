"""Tests for publishing a Library item into the Library root."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from audify.library.metadata import BookMetadata, SeriesEntry
from audify.library.publish import (
    ChapterMark,
    abs_metadata,
    audio_names,
    check_separate,
    item_relative_dir,
    publish_library_item,
    resolve_library_root,
    safe_name,
)

ABS_KEYS = {
    "tags", "title", "subtitle", "authors", "narrators", "series", "genres",
    "publishedYear", "publishedDate", "publisher", "description", "isbn",
    "asin", "language", "explicit", "abridged", "chapters",
}


def _meta(**kw: Any) -> BookMetadata:
    base: dict[str, Any] = dict(
        title="A Ordem do Capital",
        authors=["Clara E. Mattei"],
        narrators=["Kokoro (pf_dora)"],
        language="pt",
        published_date="2023-05-01",
        isbn="9786557171234",
    )
    base.update(kw)
    return BookMetadata(**base)


def _files(tmp_path: Path, *names: str) -> list[Path]:
    out = []
    for name in names:
        path = tmp_path / "work" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio:" + name.encode())
        out.append(path)
    return out


def test_item_folder_is_author_series_title():
    assert item_relative_dir(_meta()) == Path("Clara E. Mattei/A Ordem do Capital")
    with_series = _meta(series=[SeriesEntry("Ensaios", "2")])
    assert item_relative_dir(with_series) == Path(
        "Clara E. Mattei/Ensaios/A Ordem do Capital"
    )
    assert item_relative_dir(_meta(authors=[])) == Path(
        "Unknown Author/A Ordem do Capital"
    )


def test_safe_name_strips_unsafe_characters_and_long_names():
    assert safe_name('Title: "Sub" / Part?') == "Title Sub Part"
    assert safe_name("  .hidden. ") == "hidden"
    long = "á" * 200
    assert len(safe_name(long).encode()) <= 120


def test_metadata_json_uses_only_audiobookshelf_keys():
    data = abs_metadata(
        _meta(series=[SeriesEntry("Ensaios", "2")]),
        chapters=[ChapterMark("One", 0.0, 61.25), ChapterMark("Two", 61.25, 90.0)],
    )
    assert set(data) <= ABS_KEYS
    assert data["series"] == ["Ensaios #2"]
    assert data["publishedYear"] == "2023"
    assert data["isbn"] == "9786557171234"
    assert data["language"] == "pt"
    assert data["abridged"] is False
    assert data["chapters"][1] == {"id": 1, "start": 61.25, "end": 90.0, "title": "Two"}
    assert "chapters" not in abs_metadata(_meta())


def test_audio_names_for_single_file_and_parts(tmp_path):
    one = _files(tmp_path, "slug.m4b")
    assert audio_names(_meta(), one) == ["A Ordem do Capital.m4b"]
    parts = _files(tmp_path, "s_part1.m4b", "s_part2.m4b")
    assert audio_names(_meta(), parts) == [
        "A Ordem do Capital - Part 01.m4b",
        "A Ordem do Capital - Part 02.m4b",
    ]


def test_publish_creates_complete_item(tmp_path):
    root = tmp_path / "library"
    audio = _files(tmp_path, "slug.m4b")
    ebook = _files(tmp_path, "book.epub")[0]
    cover = _files(tmp_path, "cover.png")[0]

    item = publish_library_item(root, _meta(), audio, source_ebook=ebook, cover=cover)

    assert item == root / "Clara E. Mattei" / "A Ordem do Capital"
    assert sorted(p.name for p in item.iterdir()) == [
        "A Ordem do Capital.m4b",
        "book.epub",
        "cover.jpg",
        "metadata.json",
    ]
    data = json.loads((item / "metadata.json").read_text(encoding="utf-8"))
    assert data["title"] == "A Ordem do Capital"
    assert data["narrators"] == ["Kokoro (pf_dora)"]
    # Nothing but the item (and its author folder) is left in the root.
    assert [p.name for p in root.iterdir()] == ["Clara E. Mattei"]


def test_failed_publish_leaves_nothing_in_the_root(tmp_path, monkeypatch):
    root = tmp_path / "library"

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "rename", boom)
    with pytest.raises(OSError, match="disk full"):
        publish_library_item(root, _meta(), _files(tmp_path, "a.m4b"))
    assert list(root.iterdir()) == []


def test_existing_item_is_not_overwritten(tmp_path):
    root = tmp_path / "library"
    publish_library_item(root, _meta(), _files(tmp_path, "a.m4b"))
    with pytest.raises(FileExistsError):
        publish_library_item(root, _meta(), _files(tmp_path, "a.m4b"))


def test_publish_requires_audio(tmp_path):
    with pytest.raises(ValueError, match="No audio"):
        publish_library_item(tmp_path / "library", _meta(), [])


def test_library_root_must_be_separate_from_work_dir(tmp_path):
    work = tmp_path / "output"
    with pytest.raises(ValueError, match="overlaps"):
        check_separate(work, work)
    with pytest.raises(ValueError, match="overlaps"):
        check_separate(work / "library", work)
    with pytest.raises(ValueError, match="overlaps"):
        check_separate(tmp_path, work)
    check_separate(tmp_path / "library", work)


def test_library_root_from_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIFY_LIBRARY_ROOT", raising=False)
    assert resolve_library_root(None) is None
    monkeypatch.setenv("AUDIFY_LIBRARY_ROOT", str(tmp_path / "env"))
    assert resolve_library_root(None) == (tmp_path / "env").resolve()
    assert resolve_library_root(tmp_path / "x") == (tmp_path / "x").resolve()


def test_creator_publishes_after_m4b(tmp_path):
    """create_m4b's last step publishes the finished M4B with its Source ebook."""
    from audify.audiobook_creator import AudiobookCreator

    creator = object.__new__(AudiobookCreator)
    creator.library_root = tmp_path / "library"
    creator.book_metadata = _meta()
    creator.m4b_outputs = _files(tmp_path, "slug.m4b")
    creator.path = _files(tmp_path, "book.epub")[0]
    creator.cover_image_path = None

    item = creator._publish_library_item()

    assert item is not None
    assert (item / "A Ordem do Capital.m4b").exists()
    assert (item / "book.epub").exists()


def test_creator_without_library_root_publishes_nothing(tmp_path):
    from audify.audiobook_creator import AudiobookCreator

    creator = object.__new__(AudiobookCreator)
    creator.m4b_outputs = _files(tmp_path, "slug.m4b")
    assert creator._publish_library_item() is None
