from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from audify.qa.state import CycleId, FlagEntry, GraphState

logger = logging.getLogger(__name__)

# Per-cycle retry budget. Mirrors the bounded-attempts halting condition
# documented in ADR-0001 ("Each back-edge must have a bounded halting
# condition"). Each cycle owns its own counter.
MAX_BUDGET_PER_CYCLE = 3

# Canonical cycle ordering for table output. Matches the CycleId Literal in
# audify.qa.state and the schema documented in CONTEXT.md.
CYCLE_ORDER = ("cycle_1_escalation", "cycle_2_reroute", "cycle_3_retry")

_VERDICT_STYLE = {
    "clean": "green",
    "flagged": "yellow",
    "unrecoverable": "red",
}

_VERDICT_GLYPH = {
    "clean": "✓",
    "flagged": "⚠",
    "unrecoverable": "✗",
}


def report_node(state: GraphState) -> dict:
    """Aggregate per-chapter flags into the end-of-run quality report.

    Produces three artifacts:

    * a Rich-rendered table printed to stdout,
    * ``quality_report.txt`` next to the audiobook (same content as the
      Rich table, plain text),
    * ``quality_report.json`` next to the audiobook (machine-readable;
      schema documented in ``CONTEXT.md``).

    A chapter's verdict is derived from its ``FlagEntry`` list:

    * ``clean`` — no flags raised.
    * ``flagged`` — at least one flag, none exhausted (a cycle fired and
      resolved within budget).
    * ``unrecoverable`` — at least one flag with ``exhausted=True`` (a
      cycle ran out of retries; best-effort artifact kept per ``0bffc34``).

    When the creator was launched with ``--warn-stop`` and any chapter
    ended up ``unrecoverable``, the aggregator prompts the user before
    returning. The default (warn-only) path logs and continues without
    prompting, preserving the behaviour established by commit ``0bffc34``.
    """
    creator = state["creator"]
    chapter_titles = state["chapter_titles"]
    episode_paths = state.get("episode_paths", [])
    flags = state.get("flags", {})
    best_wer = state.get("best_wer", {})
    retry_budget = state.get("retry_budget", {})

    creator.progress.set_phase("Generating quality report")

    chapters_report: dict[str, dict] = {}
    for i, title in enumerate(chapter_titles, 1):
        chapter_id = f"chapter_{i}"
        chapter_flags: list[FlagEntry] = flags.get(chapter_id, [])
        chapter_budget = retry_budget.get(chapter_id, {})

        verdict = _derive_verdict(chapter_flags)
        attempts_used = _attempts_used(chapter_flags, chapter_budget)

        chapters_report[chapter_id] = {
            "title": title,
            "verdict": verdict,
            "attempts_used": attempts_used,
            "best_wer": best_wer.get(chapter_id),
            "flags": [dict(entry) for entry in chapter_flags],
        }

    pipeline_status = _derive_pipeline_status(chapter_titles, episode_paths)
    verdict_counts = _count_verdicts(chapters_report)

    report = {
        "pipeline_status": pipeline_status,
        "episodes_synthesised": len(episode_paths),
        "chapters_expected": len(chapter_titles),
        "verdict_counts": verdict_counts,
        "chapters": chapters_report,
    }

    audiobook_path = Path(creator.audiobook_path)
    audiobook_path.mkdir(parents=True, exist_ok=True)

    json_path = audiobook_path / "quality_report.json"
    json_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Quality report (JSON) written to {json_path}")

    text_path = audiobook_path / "quality_report.txt"
    rendered = _render_human_readable(
        audiobook_path.name, chapters_report, verdict_counts, pipeline_status
    )
    text_path.write_text(rendered)
    logger.info(f"Quality report (text) written to {text_path}")

    # Preserve --warn-stop semantics from commit 0bffc34: unrecoverable
    # verdicts surface as a prompt only when warn_stop is set; the default
    # path warns and continues.
    if verdict_counts.get("unrecoverable", 0) and getattr(
        creator, "warn_stop", False
    ):
        _warn_stop_prompt(verdict_counts["unrecoverable"])

    return {}


def _derive_verdict(chapter_flags: list[FlagEntry]) -> str:
    if not chapter_flags:
        return "clean"
    if any(entry["exhausted"] for entry in chapter_flags):
        return "unrecoverable"
    return "flagged"


def _attempts_used(
    chapter_flags: list[FlagEntry],
    chapter_budget: dict[CycleId, int],
) -> dict[str, int]:
    """Return attempts-used per cycle, only for cycles that actually fired.

    A cycle is considered "fired" if it has either a remaining-budget entry
    or a flag entry. Cycles that never ran are omitted from the dict rather
    than reported as ``0/3`` — keeps the report focused on what actually
    happened.

    Iterates over :data:`CYCLE_ORDER` (then any unknown cycles in sorted
    order) so the resulting dict has deterministic key insertion order,
    keeping ``quality_report.json`` diffs stable across runs.
    """
    fired_cycles = set(chapter_budget.keys()) | {
        entry["cycle"] for entry in chapter_flags
    }
    ordered_cycles = [c for c in CYCLE_ORDER if c in fired_cycles] + sorted(
        fired_cycles - set(CYCLE_ORDER)
    )
    return {
        cycle: MAX_BUDGET_PER_CYCLE - chapter_budget.get(cycle, MAX_BUDGET_PER_CYCLE)
        for cycle in ordered_cycles
    }


def _derive_pipeline_status(
    chapter_titles: list[str],
    episode_paths: list,
) -> str:
    expected = len(chapter_titles)
    produced = len(episode_paths)
    if expected == 0:
        return "no_chapters"
    if produced == expected:
        return "complete"
    if produced == 0:
        return "failed"
    return "partial"


def _count_verdicts(chapters_report: dict[str, dict]) -> dict[str, int]:
    counts = {"clean": 0, "flagged": 0, "unrecoverable": 0}
    for entry in chapters_report.values():
        counts[entry["verdict"]] += 1
    return counts


def _render_human_readable(
    audiobook_name: str,
    chapters_report: dict[str, dict],
    verdict_counts: dict[str, int],
    pipeline_status: str,
) -> str:
    """Render the report to stdout and return the captured plain-text form."""
    console = Console(record=True)

    table = Table(title=f"Quality Report — {audiobook_name}")
    table.add_column("Chapter", overflow="fold")
    table.add_column("Verdict")
    table.add_column("Details", overflow="fold")

    for chapter_id, entry in chapters_report.items():
        verdict = entry["verdict"]
        glyph = _VERDICT_GLYPH.get(verdict, " ")
        style = _VERDICT_STYLE.get(verdict, "")
        table.add_row(
            f"{chapter_id} — {entry['title']}",
            f"{glyph} {verdict}",
            _format_details(entry),
            style=style,
        )

    console.print(table)
    summary = (
        f"Summary: {verdict_counts['clean']} clean · "
        f"{verdict_counts['flagged']} flagged · "
        f"{verdict_counts['unrecoverable']} unrecoverable · "
        f"{len(chapters_report)} expected · "
        f"pipeline_status={pipeline_status}"
    )
    console.print(summary)
    return console.export_text()


def _format_details(entry: dict) -> str:
    flags: list[dict] = entry["flags"]
    if not flags:
        return ""

    best_wer: Optional[float] = entry.get("best_wer")
    attempts: dict[str, int] = entry["attempts_used"]
    parts: list[str] = []
    for flag in flags:
        cycle = flag["cycle"]
        used = attempts.get(cycle, 0)
        if flag["exhausted"]:
            tail = f"{used}/{MAX_BUDGET_PER_CYCLE} attempts exhausted"
        else:
            tail = "resolved"
        suffix = f" — {flag['reason']}" if flag.get("reason") else ""
        if cycle == "cycle_3_retry" and best_wer is not None:
            tail = f"{tail}, best WER {best_wer:.2f}"
        parts.append(f"{cycle} ({tail}){suffix}")
    return "; ".join(parts)


def _warn_stop_prompt(unrecoverable_count: int) -> None:
    """Prompt the user when warn-stop is set and unrecoverable flags exist.

    No-ops the prompt on EOFError / OSError (non-tty environments such as
    CI captures), so the warn-stop guard never hangs the run.
    """
    message = (
        f"\n⚠️  Quality report flagged {unrecoverable_count} chapter"
        f"{'s' if unrecoverable_count != 1 else ''} as unrecoverable. "
        "Accept anyway? (y/N): "
    )
    try:
        response = input(message)
    except (EOFError, OSError):
        logger.warning(
            "warn-stop set but no interactive stdin available; "
            "continuing without prompt."
        )
        return
    if response.lower() in ("y", "yes"):
        logger.info("User accepted audiobook despite unrecoverable chapters.")
    else:
        logger.warning(
            "User did not accept audiobook; "
            "best-effort artifact kept on disk, review flagged chapters."
        )
