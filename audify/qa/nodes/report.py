from __future__ import annotations

import json
import logging
from pathlib import Path

from audify.qa.state import GraphState

logger = logging.getLogger(__name__)


def report_node(state: GraphState) -> dict:
    """Write a JSON quality report alongside the output directory.

    Each chapter entry records the flags raised during the run, the number of
    retry attempts consumed, and the best WER seen (populated by future
    cycle nodes; empty in the linear skeleton).
    """
    creator = state["creator"]
    chapter_titles = state["chapter_titles"]
    episode_paths = state.get("episode_paths", [])
    flags = state.get("flags", {})
    best_wer = state.get("best_wer", {})
    retry_budget = state.get("retry_budget", {})

    max_budget = 3
    chapters_report: dict[str, dict] = {}
    for i, title in enumerate(chapter_titles, 1):
        chapter_id = f"chapter_{i}"
        budget_remaining = retry_budget.get(chapter_id, max_budget)
        # Until the cyclic detectors land, no node writes to `flags` /
        # `best_wer` / `retry_budget`. Surface that explicitly instead of
        # reporting a misleading "ok" for every chapter.
        chapter_flags = flags.get(chapter_id)
        if chapter_flags:
            verdict = "flagged"
        elif chapter_id in best_wer or chapter_id in retry_budget:
            verdict = "ok"
        else:
            verdict = "skeleton"
        chapters_report[chapter_id] = {
            "title": title,
            "verdict": verdict,
            "attempts_used": max_budget - budget_remaining,
            "best_wer": best_wer.get(chapter_id),
            "flags": flags.get(chapter_id, []),
        }

    expected_chapters = len(chapter_titles)
    if expected_chapters == 0:
        pipeline_status = "no_chapters"
    elif len(episode_paths) == expected_chapters:
        pipeline_status = "complete"
    elif len(episode_paths) == 0:
        pipeline_status = "failed"
    else:
        pipeline_status = "partial"

    report = {
        "pipeline_status": pipeline_status,
        "episodes_synthesised": len(episode_paths),
        "chapters_expected": expected_chapters,
        "chapters": chapters_report,
    }

    report_path = Path(creator.audiobook_path) / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Quality report written to {report_path}")

    return {}
