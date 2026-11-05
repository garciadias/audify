"""Tests for rendition kinds: Book renditions, Derived works, translations."""

import json
from pathlib import Path

import pytest

from audify.library.identity import ItemIdentity
from audify.library.metadata import BookMetadata
from audify.library.publish import publish_library_item
from audify.library.rendition import LLM_SCRIPT_TAG, apply_rendition, rendition_kind


def _meta() -> BookMetadata:
    return BookMetadata(
        title="The Three-Body Problem", authors=["Liu Cixin"], language="en"
    )


def test_audiobook_task_is_a_book_rendition_with_llm_script_tag():
    r = apply_rendition(_meta(), task="audiobook", requires_llm=True)
    assert r.metadata.title == "The Three-Body Problem"
    assert r.metadata.tags == [LLM_SCRIPT_TAG]
    assert r.include_ebook is True


def test_direct_task_is_verbatim_and_untagged():
    r = apply_rendition(_meta(), task="direct", requires_llm=False)
    assert r.metadata.tags == []
    assert r.include_ebook is True


@pytest.mark.parametrize("task", ["podcast", "summary", "lecture", "meditation"])
def test_other_tasks_are_derived_works(task):
    r = apply_rendition(_meta(), task=task, requires_llm=True)
    assert r.metadata.title == f"The Three-Body Problem ({task.capitalize()})"
    assert r.metadata.tags == [task]
    assert LLM_SCRIPT_TAG not in r.metadata.tags
    assert r.include_ebook is False


def test_translated_rendition_keeps_title_and_changes_language():
    r = apply_rendition(_meta(), task="audiobook", requires_llm=True, translate_to="pt")
    assert r.metadata.title == "The Three-Body Problem"
    assert r.metadata.language == "pt"
    assert "translated-from-en" in r.metadata.tags
    assert r.folder_title == "The Three-Body Problem [pt]"
    assert r.include_ebook is True


def test_translation_to_the_same_language_is_not_a_translation():
    r = apply_rendition(_meta(), task="audiobook", requires_llm=True, translate_to="en")
    assert r.folder_title == "The Three-Body Problem"
    assert not any(t.startswith("translated-from") for t in r.metadata.tags)


def test_source_metadata_is_not_modified():
    meta = _meta()
    apply_rendition(meta, task="podcast", requires_llm=True, translate_to="pt")
    assert meta.title == "The Three-Body Problem"
    assert meta.tags == []
    assert meta.language == "en"


def _audio(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "work" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    return path


def _publish(root, tmp_path, ebook, task, llm, lang):
    r = apply_rendition(_meta(), task=task, requires_llm=llm, translate_to=lang)
    return publish_library_item(
        root,
        r.metadata,
        [_audio(tmp_path, f"{task}-{lang}.m4b")],
        source_ebook=ebook if r.include_ebook else None,
        identity=ItemIdentity("v1:same-book", task=rendition_kind(task), language=lang),
        folder_title=r.folder_title,
    )


def test_renditions_of_one_book_are_separate_items(tmp_path):
    root = tmp_path / "library"
    ebook = _audio(tmp_path, "book.epub")
    book = _publish(root, tmp_path, ebook, "audiobook", True, None)
    podcast = _publish(root, tmp_path, ebook, "podcast", True, None)
    translated = _publish(root, tmp_path, ebook, "audiobook", True, "pt")

    assert len({book, podcast, translated}) == 3
    author = root / "Liu Cixin"
    assert book == author / "The Three-Body Problem"
    assert podcast == author / "The Three-Body Problem (Podcast)"
    assert translated == author / "The Three-Body Problem [pt]"
    assert not list(podcast.glob("*.epub"))
    assert (translated / "book.epub").exists()


def test_direct_after_audiobook_republishes_the_same_item(tmp_path):
    root = tmp_path / "library"
    ebook = _audio(tmp_path, "book.epub")
    first = _publish(root, tmp_path, ebook, "audiobook", True, None)
    second = _publish(root, tmp_path, ebook, "direct", False, None)

    assert second == first
    data = json.loads((first / "metadata.json").read_text())
    assert LLM_SCRIPT_TAG not in data["tags"]
    assert (first / "The Three-Body Problem.m4b").read_bytes() == b"direct-None.m4b"


def test_rendition_kind():
    assert rendition_kind("audiobook") == "book"
    assert rendition_kind("direct") == "book"
    assert rendition_kind("Podcast") == "podcast"
