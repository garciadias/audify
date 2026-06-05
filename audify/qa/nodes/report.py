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
    flags = state.get("flags", {})
    best_wer = state.get("best_wer", {})
    retry_budget = state.get("retry_budget", {})

    max_budget = 3
    report: dict[str, dict] = {}
    for i, title in enumerate(chapter_titles, 1):
        chapter_id = f"chapter_{i}"
        budget_remaining = retry_budget.get(chapter_id, max_budget)
        report[chapter_id] = {
            "title": title,
            "verdict": "flagged" if flags.get(chapter_id) else "ok",
            "attempts_used": max_budget - budget_remaining,
            "best_wer": best_wer.get(chapter_id),
            "flags": flags.get(chapter_id, []),
        }

    report_path = Path(creator.audiobook_path) / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Quality report written to {report_path}")

    return {}
