from __future__ import annotations

from pathlib import Path
from typing import Any

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """Shared state carried by the LangGraph QA pipeline between nodes.

    Fields prefixed with ``retry_budget``, ``best_wer``, and ``flags`` are
    keyed by ``chapter_id`` (``"chapter_{episode_number}"``).  They are
    populated as the graph processes each chapter and are consumed by the
    report node at the end.
    """

    creator: Any  # AudiobookCreator instance — not serialised
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
