"""Error and edge branches of library publishing, overrides and catalogues."""

import io
import json
import logging
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from audify.audiobook_creator import AudiobookCreator
from audify.library import catalogue
from audify.library.catalogue import (
    CatalogueRecord,
    GoogleBooks,
    OpenLibrary,
    default_catalogues,
    enrich_from_catalogues,
    fetch_cover,
    http_fetch,
    lookup_enabled,
)
from audify.library.identity import ItemIdentity
from audify.library.metadata import BookMetadata, SeriesEntry
from audify.library.override import apply_override, load_override
from audify.library.publish import (
    MANIFEST,
    find_library_item,
    publish_library_item,
    read_manifest,
)
from audify.readers.pdf import PdfReader


def _file(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _meta(**kw: Any) -> BookMetadata:
    base: dict[str, Any] = {"title": "T", "authors": ["A"]}
    base.update(kw)
    return BookMetadata(**base)


# --- publish -------------------------------------------------------------


def test_no_audio_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No audio"):
        publish_library_item(tmp_path, _meta(), [])


def test_read_manifest_missing_broken_or_not_an_object(tmp_path):
    assert read_manifest(tmp_path) is None
    (tmp_path / MANIFEST).write_text("{broken")
    assert read_manifest(tmp_path) is None
    (tmp_path / MANIFEST).write_text("[1, 2]")
    assert read_manifest(tmp_path) is None


def test_find_skips_hidden_folders_and_manifests_without_identity(tmp_path):
    identity = ItemIdentity("v1:k")
    hidden = tmp_path / ".audify-publishing-x"
    _file(hidden / MANIFEST, json.dumps({"identity": identity.to_json()}).encode())
    _file(tmp_path / "A" / "T" / MANIFEST, json.dumps({"identity": None}).encode())
    assert find_library_item(tmp_path, identity) is None
    assert find_library_item(tmp_path / "missing", identity) is None


def test_republish_keeps_files_audify_did_not_write(tmp_path):
    root = tmp_path / "lib"
    identity = ItemIdentity("v1:k")
    item = publish_library_item(
        root, _meta(), [_file(tmp_path / "a.m4b", b"1")], identity=identity
    )
    # The user drops their own cover in; Audify now also has one to publish.
    (item / "cover.jpg").write_bytes(b"mine")
    publish_library_item(
        root,
        _meta(),
        [_file(tmp_path / "b.m4b", b"2")],
        cover=_file(tmp_path / "c.jpg", b"audify"),
        identity=identity,
    )
    assert (item / "cover.jpg").read_bytes() == b"mine"


def test_failed_publish_tolerates_a_parent_that_cannot_be_removed(tmp_path):
    root = tmp_path / "lib"
    with (
        patch("audify.library.publish.os.rename", side_effect=OSError("boom")),
        patch.object(Path, "rmdir", side_effect=OSError("busy")),
        pytest.raises(OSError, match="boom"),
    ):
        publish_library_item(root, _meta(), [_file(tmp_path / "a.m4b")])
    assert not list(root.glob(".audify-publishing-*"))


# --- override ------------------------------------------------------------


def test_override_that_is_not_an_object_is_ignored(tmp_path):
    path = _file(tmp_path / "o.json", b'["title"]')
    assert load_override(path) is None


def test_override_series_forms_and_cleared_lists():
    meta = apply_override(
        _meta(genres=["G"]),
        {
            "series": [{"name": "Saga", "sequence": 3}, "Plain", {"nope": 1}],
            "genres": None,
        },
    )
    assert meta.series == [SeriesEntry("Saga", "3"), SeriesEntry("Plain")]
    assert meta.genres == []


def test_override_single_series_string():
    meta = apply_override(_meta(), {"series": "Saga #1.5"})
    assert meta.series == [SeriesEntry("Saga", "1.5")]


# --- catalogue -----------------------------------------------------------


def test_http_fetch_returns_body(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"ok"
    monkeypatch.setattr(catalogue.urllib.request, "urlopen", lambda *a, **k: response)
    assert http_fetch("https://example.org") == b"ok"


@pytest.mark.parametrize(
    ("error", "logged"),
    [
        (urllib.error.HTTPError("u", 429, "slow", {}, io.BytesIO()), True),  # type: ignore[arg-type]
        (urllib.error.HTTPError("u", 404, "gone", {}, io.BytesIO()), False),  # type: ignore[arg-type]
        (urllib.error.URLError("no network"), True),
        (TimeoutError(), True),
    ],
)
def test_http_fetch_never_raises(monkeypatch, caplog, error, logged):
    def boom(*a, **k):
        raise error

    monkeypatch.setattr(catalogue.urllib.request, "urlopen", boom)
    with caplog.at_level(logging.WARNING):
        assert http_fetch("https://example.org") is None
    assert bool(caplog.records) is logged


def test_bad_json_is_no_match():
    assert OpenLibrary(lambda u: b"<html>").by_isbn("1") is None
    assert OpenLibrary(lambda u: b"[1]").by_isbn("1") is None


def test_lookup_switch_and_google_key(monkeypatch):
    monkeypatch.setenv("AUDIFY_METADATA_LOOKUP", "off")
    assert not lookup_enabled()
    monkeypatch.setenv("AUDIFY_METADATA_LOOKUP", "1")
    assert lookup_enabled()
    monkeypatch.delenv("GOOGLE_BOOKS_API_KEY", raising=False)
    assert [c.name for c in default_catalogues()] == ["openlibrary"]
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "k")
    assert [c.name for c in default_catalogues()] == ["openlibrary", "googlebooks"]


def test_google_title_search_drops_edition_facts():
    body = json.dumps(
        {
            "items": [
                {"volumeInfo": {"title": "Other", "authors": ["A"]}},
                {
                    "volumeInfo": {
                        "title": "T: a novel",
                        "authors": ["Ann A"],
                        "publisher": "P",
                        "categories": ["Fiction", "Ukraine -- Fiction"],
                        "description": "D",
                        "industryIdentifiers": [
                            {"type": "ISBN_13", "identifier": "978"}
                        ],
                    }
                },
            ]
        }
    ).encode()
    record = GoogleBooks("k", lambda u: body).by_title("T", ["A"])
    assert record is not None
    assert record.genres == ["Fiction"]
    assert record.description == "D"
    assert record.isbn == ""
    assert record.publisher == ""
    assert GoogleBooks("k", lambda u: b"{}").by_title("T", []) is None
    assert GoogleBooks("k", lambda u: b"{}").by_isbn("1") is None


def test_isbn_record_from_google_reads_isbn13():
    body = json.dumps(
        {
            "items": [
                {
                    "volumeInfo": {
                        "industryIdentifiers": [
                            {"type": "ISBN_10", "identifier": "10"},
                            {"type": "ISBN_13", "identifier": "13"},
                        ]
                    }
                }
            ]
        }
    ).encode()
    gb = GoogleBooks("k", lambda u: body)
    assert gb._record(json.loads(body)["items"][0]["volumeInfo"]).isbn == "13"


def test_open_library_without_work_or_author_names():
    edition = json.dumps(
        {"title": "T", "authors": [{"key": "/authors/X"}, "junk"], "works": []}
    ).encode()

    def fetch(url):
        return edition if "/isbn/" in url else b"{}"

    record = OpenLibrary(fetch).by_isbn("1")
    assert record is not None
    assert record.authors == []
    assert record.publisher == ""


def test_genre_cleanup_skips_non_strings_and_caps():
    subjects = [None, {"name": "Dict Genre"}, *[f"G{i}" for i in range(10)]]
    assert catalogue._clean_genres(subjects) == ["Dict Genre", "G0", "G1", "G2", "G3"]


def test_catalogue_exception_does_not_fail(caplog):
    broken = MagicMock()
    broken.name = "broken"
    broken.by_isbn.side_effect = RuntimeError("bug")
    meta = _meta(isbn="1")
    with caplog.at_level(logging.WARNING):
        assert enrich_from_catalogues(meta, [broken]) == []
    assert "broken lookup failed" in caplog.text


def test_no_title_no_isbn_and_complete_metadata():
    ol = MagicMock()
    ol.name = "ol"
    assert enrich_from_catalogues(BookMetadata(), [ol]) == []
    ol.by_isbn.assert_not_called()
    full = _meta(
        subtitle="s",
        genres=["g"],
        published_date="d",
        publisher="p",
        description="x",
        isbn="1",
    )
    assert enrich_from_catalogues(full, [ol]) == []
    ol.by_isbn.assert_not_called()


def test_fetch_cover_skips_records_without_url(tmp_path):
    assert fetch_cover([CatalogueRecord("x")], tmp_path / "c.jpg") is None


# --- AudiobookCreator wiring ----------------------------------------------


def _creator(tmp_path: Path) -> AudiobookCreator:
    creator = object.__new__(AudiobookCreator)
    creator.library_root = tmp_path / "lib"
    creator.audiobook_path = tmp_path / "work"
    creator.audiobook_path.mkdir()
    creator.path = _file(tmp_path / "book.epub")
    creator.book_metadata = _meta()
    creator.cover_image_path = None
    creator.m4b_outputs = [_file(tmp_path / "work" / "b.m4b")]
    return creator


def test_publish_without_outputs_or_root(tmp_path, caplog):
    creator = _creator(tmp_path)
    creator.m4b_outputs = []
    with caplog.at_level(logging.ERROR):
        assert creator._publish_library_item() is None
    assert "nothing published" in caplog.text
    creator.library_root = None
    assert creator._publish_library_item() is None


def test_publish_refuses_a_foreign_folder(tmp_path, caplog):
    creator = _creator(tmp_path)
    (creator.library_root / "A" / "T").mkdir(parents=True)
    with (
        patch.object(AudiobookCreator, "_item_identity", return_value=None),
        caplog.at_level(logging.ERROR),
    ):
        assert creator._publish_library_item() is None
    assert "Not overwriting" in caplog.text


def test_publish_failure_is_logged_not_raised(tmp_path, caplog):
    creator = _creator(tmp_path)
    with (
        patch(
            "audify.audiobook_creator.publish_library_item",
            side_effect=RuntimeError("disk full"),
        ),
        patch.object(AudiobookCreator, "_item_identity", return_value=None),
        caplog.at_level(logging.ERROR),
    ):
        assert creator._publish_library_item() is None
    assert "disk full" in caplog.text


def test_item_identity_for_pdf_and_on_failure(tmp_path):
    creator = _creator(tmp_path)
    creator.task_name = "podcast"
    creator.translate = "pt"
    creator.reader = MagicMock(spec=PdfReader)
    creator.reader.cleaned_text = "some text"
    identity = creator._item_identity()
    assert identity is not None
    assert identity.task == "podcast"
    assert identity.language == "pt"

    creator.reader = MagicMock()  # not a PDF, and the "EPUB" is not a zip
    assert creator._item_identity() is None


def test_chapter_list_skips_bad_and_empty_episodes(tmp_path):
    creator = _creator(tmp_path)
    episodes = tmp_path / "episodes"
    for name in ["episode_001.mp3", "episode_002.mp3", "episode_x.mp3"]:
        _file(episodes / name)
    creator.episodes_path = episodes
    creator.chapter_titles = []
    durations = {"episode_001.mp3": 5.0, "episode_002.mp3": 0.0}
    with patch(
        "audify.audiobook_creator.AudioProcessor.get_duration",
        side_effect=lambda p: durations[Path(p).name],
    ):
        marks = creator._whole_book_chapters()
    assert [(m.title, m.start, m.end) for m in marks] == [("Episode 1", 0.0, 5.0)]
    creator.episodes_path = None  # type: ignore[assignment]
    assert creator._whole_book_chapters() == []


def test_enrichment_runs_only_when_publishing(tmp_path, monkeypatch):
    creator = _creator(tmp_path)
    monkeypatch.setenv("AUDIFY_METADATA_LOOKUP", "1")
    record = CatalogueRecord("openlibrary", cover_url="u")

    def fake_enrich(meta):
        meta.publisher = "P"
        return [record]

    cover = tmp_path / "work" / "cover.catalogue.jpg"
    with (
        patch("audify.audiobook_creator.enrich_from_catalogues", fake_enrich),
        patch("audify.audiobook_creator.fetch_cover", return_value=cover),
    ):
        meta = _meta()
        creator._enrich_from_catalogues(meta)
    assert meta.publisher == "P"
    assert creator.cover_image_path == cover
    assert meta.provenance["cover"] == "openlibrary"

    creator.library_root = None
    with patch("audify.audiobook_creator.enrich_from_catalogues") as enrich:
        creator._enrich_from_catalogues(_meta())
    enrich.assert_not_called()
