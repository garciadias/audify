"""Rendition kinds: what a Library item is relative to its book.

* A Book rendition (``audiobook`` or ``direct`` task) is the book itself and
  holds the Source ebook. An ``audiobook`` rendition speaks an LLM Script, so it
  carries the ``llm-script`` tag; ``direct`` is verbatim and carries none.
* A Derived work (``podcast``, ``summary``, ``lecture``, ``meditation``, or any
  other task) is its own item titled ``Title (Task)``, tagged with the task,
  with no ebook, so it never merges into the book's item.
* A Translated rendition keeps the ebook's title, takes the target language,
  is tagged ``translated-from-<lang>``, and lives in ``Title [<lang>]``.

See CONTEXT.md, "Library delivery".
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from audify.library.metadata import BookMetadata

BOOK_TASKS = frozenset({"audiobook", "direct"})
LLM_SCRIPT_TAG = "llm-script"


def rendition_kind(task: str) -> str:
    """The rendition kind used in the Item identity.

    ``audiobook`` and ``direct`` are both the book itself, so they share the
    kind ``book``: a book has one Book rendition per language, and running the
    other task re-publishes that item instead of creating a second one.
    """
    task = (task or "audiobook").lower()
    return "book" if task in BOOK_TASKS else task


@dataclass(frozen=True)
class Rendition:
    """Book metadata adjusted for one rendition, plus how to publish it."""

    metadata: BookMetadata
    folder_title: str
    include_ebook: bool


def _add_tag(tags: list[str], tag: str) -> list[str]:
    return tags if tag in tags else [*tags, tag]


def apply_rendition(
    metadata: BookMetadata,
    task: str,
    requires_llm: bool,
    translate_to: Optional[str] = None,
) -> Rendition:
    """Adjust *metadata* for the rendition produced by *task*.

    Args:
        metadata: Book metadata of the source book (not modified).
        task: The task name that produced the audio.
        requires_llm: Whether the audio speaks an LLM-rewritten Script.
        translate_to: The translation target language, if the audio was
            translated.
    """
    task = (task or "audiobook").lower()
    tags = list(metadata.tags)
    title = metadata.title
    is_book = task in BOOK_TASKS

    if is_book:
        if requires_llm:
            tags = _add_tag(tags, LLM_SCRIPT_TAG)
    else:
        title = f"{metadata.title} ({task.capitalize()})"
        tags = _add_tag(tags, task)

    language = metadata.language
    folder_title = title
    if translate_to and translate_to != metadata.language:
        if metadata.language:
            tags = _add_tag(tags, f"translated-from-{metadata.language}")
        language = translate_to
        folder_title = f"{title} [{translate_to}]"

    adjusted = dataclasses.replace(metadata, title=title, tags=tags, language=language)
    return Rendition(
        metadata=adjusted, folder_title=folder_title, include_ebook=is_book
    )
