"""Tests for the Book key, the Item manifest, and Re-publish."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from ebooklib import epub

from audify.library.identity import (
    KEY_VERSION,
    ItemIdentity,
    book_key_from_epub,
    book_key_from_text,
)
from audify.library.metadata import BookMetadata
from audify.library.publish import MANIFEST, find_library_item, publish_library_item


def _write_epub(path: Path, title: str, author: str, text: str) -> Path:
    book = epub.EpubBook()
    book.set_identifier("urn:uuid:fixed")
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)
    chapter = epub.EpubHtml(title="One", file_name="one.xhtml", lang="en")
    chapter.content = f"<h1>One</h1><p>{text}</p>"
    book.add_item(chapter)
    book.toc = [chapter]
    book.spine = ["nav", chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


def _file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _meta(**kw: Any) -> BookMetadata:
    base: dict[str, Any] = {"title": "The Book", "authors": ["Ann Author"]}
    base.update(kw)
    return BookMetadata(**base)


def test_metadata_edit_keeps_the_book_key(tmp_path):
    a = _write_epub(tmp_path / "a.epub", "Old Title", "Old Author", "Same text.")
    b = _write_epub(tmp_path / "b.epub", "New Title", "New Author", "Same text.")
    assert a.read_bytes() != b.read_bytes()
    assert book_key_from_epub(a) == book_key_from_epub(b)


def test_different_text_changes_the_book_key(tmp_path):
    a = _write_epub(tmp_path / "a.epub", "T", "A", "First edition text.")
    b = _write_epub(tmp_path / "b.epub", "T", "A", "Revised edition text.")
    assert book_key_from_epub(a) != book_key_from_epub(b)


def test_book_key_is_deterministic_and_versioned(tmp_path):
    a = _write_epub(tmp_path / "a.epub", "T", "A", "Text.")
    assert book_key_from_epub(a) == book_key_from_epub(a)
    assert book_key_from_epub(a).startswith(f"v{KEY_VERSION}:")
    assert book_key_from_text("a  b\n c") == book_key_from_text("a b c")


@pytest.fixture
def first_publish(tmp_path):
    root = tmp_path / "library"
    ebook = _file(tmp_path / "work" / "book.epub", b"ebook")
    cover = _file(tmp_path / "work" / "cover.jpg", b"cover-1")
    audio = _file(tmp_path / "work" / "book.m4b", b"audio-1")
    identity = ItemIdentity(book_key="v1:abc")
    item = publish_library_item(
        root, _meta(), [audio], source_ebook=ebook, cover=cover, identity=identity
    )
    return root, item, identity, tmp_path


def test_manifest_records_identity_folder_and_hashes(first_publish):
    root, item, identity, _ = first_publish
    manifest = json.loads((item / MANIFEST).read_text())
    assert manifest["identity"]["book_key"] == "v1:abc"
    assert manifest["identity"]["key_version"] == KEY_VERSION
    assert manifest["folder"] == "Ann Author/The Book"
    assert manifest["ebook"] == "book.epub"
    assert set(manifest["files"]) == {
        "The Book.m4b",
        "book.epub",
        "cover.jpg",
        "metadata.json",
    }
    assert find_library_item(root, identity) == item


def test_republish_keeps_the_folder_when_title_and_author_change(first_publish):
    root, item, identity, tmp_path = first_publish
    audio = _file(tmp_path / "work2" / "book.m4b", b"audio-2")
    again = publish_library_item(
        root,
        _meta(title="Renamed", authors=["Clara E. Mattei"]),
        [audio],
        source_ebook=tmp_path / "work" / "book.epub",
        identity=identity,
    )
    assert again == item
    data = json.loads((item / "metadata.json").read_text())
    assert data["title"] == "Renamed"
    assert data["authors"] == ["Clara E. Mattei"]
    assert (item / "Renamed.m4b").read_bytes() == b"audio-2"
    # The old audio file name was Audify's and unmodified, so it is gone.
    assert not (item / "The Book.m4b").exists()
    # No second item anywhere in the library.
    assert len(list(root.rglob(MANIFEST))) == 1
    assert not (root / "Clara E. Mattei").exists()


def test_hand_replaced_cover_survives_republish(first_publish, caplog):
    root, item, identity, tmp_path = first_publish
    (item / "cover.jpg").write_bytes(b"my-own-cover")
    new_cover = _file(tmp_path / "work2" / "cover.jpg", b"cover-2")
    with caplog.at_level(logging.WARNING):
        publish_library_item(
            root,
            _meta(),
            [_file(tmp_path / "work2" / "book.m4b", b"audio-2")],
            source_ebook=tmp_path / "work" / "book.epub",
            cover=new_cover,
            identity=identity,
        )
    assert (item / "cover.jpg").read_bytes() == b"my-own-cover"
    assert "cover.jpg" in caplog.text
    assert "edited by hand" in caplog.text
    manifest = json.loads((item / MANIFEST).read_text())
    assert "cover.jpg" not in manifest["files"]  # it is the user's from now on


def test_unmodified_files_are_replaced(first_publish):
    root, item, identity, tmp_path = first_publish
    publish_library_item(
        root,
        _meta(),
        [_file(tmp_path / "work2" / "book.m4b", b"audio-2")],
        source_ebook=tmp_path / "work" / "book.epub",
        cover=_file(tmp_path / "work2" / "cover.jpg", b"cover-2"),
        identity=identity,
    )
    assert (item / "The Book.m4b").read_bytes() == b"audio-2"
    assert (item / "cover.jpg").read_bytes() == b"cover-2"


def test_republish_never_deletes_the_source_ebook(first_publish):
    root, item, identity, tmp_path = first_publish
    publish_library_item(
        root,
        _meta(),
        [_file(tmp_path / "work2" / "book.m4b", b"audio-2")],
        source_ebook=None,
        identity=identity,
    )
    assert (item / "book.epub").read_bytes() == b"ebook"


def test_files_audify_did_not_write_are_untouched(first_publish):
    root, item, identity, tmp_path = first_publish
    (item / "notes.txt").write_text("mine")
    publish_library_item(
        root,
        _meta(),
        [_file(tmp_path / "work2" / "book.m4b", b"audio-2")],
        identity=identity,
    )
    assert (item / "notes.txt").read_text() == "mine"


def test_other_rendition_is_a_separate_item(first_publish):
    root, item, identity, tmp_path = first_publish
    podcast = ItemIdentity(book_key=identity.book_key, task="podcast")
    other = publish_library_item(
        root,
        _meta(title="The Book (Podcast)"),
        [_file(tmp_path / "work3" / "p.m4b", b"pod")],
        identity=podcast,
    )
    assert other != item
    assert find_library_item(root, podcast) == other
    assert find_library_item(root, identity) == item


def test_folder_without_manifest_is_never_overwritten(tmp_path):
    root = tmp_path / "library"
    (root / "Ann Author" / "The Book").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        publish_library_item(
            root,
            _meta(),
            [_file(tmp_path / "work" / "a.m4b", b"a")],
            identity=ItemIdentity(book_key="v1:new"),
        )
