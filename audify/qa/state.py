from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from typing_extensions import TypeAlias, TypedDict

if TYPE_CHECKING:
    # Only imported for type-checkers; the runtime field is kept as ``Any``
    # because LangGraph resolves TypedDict annotations via
    # ``typing.get_type_hints``, which would fail on a forward reference to
    # ``AudiobookCreator`` (importing it at runtime here causes a cycle).
    from audify.audiobook_creator import AudiobookCreator, DirectoryAudiobookCreator

    CreatorT: TypeAlias = Union[AudiobookCreator, DirectoryAudiobookCreator]
else:
    CreatorT = Any


class GraphState(TypedDict):
    """Shared state carried by the LangGraph QA pipeline between nodes.

    Fields prefixed with ``retry_budget``, ``best_wer``, and ``flags`` are
    keyed by ``chapter_id`` (``"chapter_{episode_number}"``).  They are
    populated as the graph processes each chapter and are consumed by the
    report node at the end.
    """

    # ``CreatorT`` is ``Any`` at runtime (LangGraph evaluates this) and
    # ``AudiobookCreator | DirectoryAudiobookCreator`` under mypy, so node
    # bodies still get attribute-level type checking when annotated against
    # this state.
    creator: CreatorT
    chapters: list[str]
    chapter_titles: list[str]
    # list of (episode_number, script) pairs produced by script_gen
    chapter_scripts: list[tuple[int, str]]
    episode_paths: list[Path]
    # retry_budget[chapter_id] = remaining attempts (starts at 3 per cycle)
    retry_budget: dict[str, int]
    # best_wer[chapter_id] = lowest WER seen across retry attempts
    best_wer: dict[str, float]
    # flags[chapter_id] = human-readable flag strings for the quality report
    flags: dict[str, list[str]]
