"""Cycle-3 fidelity detector node — boundary-sampling + re-chunk retry edge.

After ``synthesize`` produces episode audio, this node transcribes a short head
window and a short tail window of each freshly-synthesized episode through the STT
client (issue #36) and fuzzy-matches each transcript against the opening / closing
words of the *script* that was fed to TTS. A high WER on either window — or a
duration ratio that corroborates a shorter-than-expected episode — flags the chapter
as a suspected TTS truncation (``CONTEXT.md`` → *Fidelity check*, *Boundary-sampling*).

Flagged chapters with remaining budget are scheduled for re-synthesis on a smaller
batch size (the only permitted remediation — *Retry edge*). The loop is bounded to
``MAX_BUDGET_PER_CYCLE`` retries per chapter; on exhaustion the lowest-WER attempt is
kept on disk and an ``exhausted=True`` flag is written for the report (#6). The book
is never aborted.

The node self-disables (passes straight through to ``assemble``) when the creator's
``fidelity_check`` flag is falsy or no STT service is reachable, so existing graph
runs without an STT service behave exactly as before.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from audify.qa.nodes.report import MAX_BUDGET_PER_CYCLE
from audify.qa.state import CycleId, FlagEntry, GraphState
from audify.qa.stt import STTClient, STTServiceError, WhisperSTTClient
from audify.qa.wer import comparable_reference, word_error_rate

logger = logging.getLogger(__name__)

# Detection thresholds (overridable via creator attributes of the same name).
WER_THRESHOLD = 0.3
FIDELITY_WINDOW_S = 8.0
DURATION_RATIO_THRESHOLD = 0.5

# Duration estimate basis — matches audify/verify.py's narration heuristic.
WORDS_PER_MINUTE = 75

_CYCLE: CycleId = "cycle_3_retry"


def fidelity_node(state: GraphState) -> dict:
    """Detect truncated episodes and schedule re-chunk retries.

    Returns a dict of updated state channels. ``pending_retry`` drives the
    conditional back-edge: non-empty → loop back to ``synthesize``.
    """
    creator = state["creator"]

    if not getattr(creator, "fidelity_check", False):
        return {}

    client = _get_stt_client(creator)
    if client is None:
        logger.info("Fidelity check skipped: no STT client available.")
        return {}

    threshold = float(getattr(creator, "wer_threshold", WER_THRESHOLD))
    window_s = float(getattr(creator, "fidelity_window_s", FIDELITY_WINDOW_S))
    ratio_threshold = float(
        getattr(creator, "duration_ratio_threshold", DURATION_RATIO_THRESHOLD)
    )
    language = getattr(creator, "resolved_language", None) or getattr(
        creator, "language", None
    )

    scripts: dict[int, str] = dict(state["chapter_scripts"])
    to_check: list[int] = state.get("episodes_to_check", [])

    # Copy nested containers so we never mutate the inbound state in place.
    retry_budget = {k: dict(v) for k, v in state.get("retry_budget", {}).items()}
    best_wer = dict(state.get("best_wer", {}))
    flags = {k: list(v) for k, v in state.get("flags", {}).items()}

    pending_retry: list[int] = []

    for episode_number in to_check:
        script = scripts.get(episode_number)
        if script is None:
            continue

        chapter_id = f"chapter_{episode_number}"
        episode_path = _episode_path(creator, episode_number)
        if not episode_path.exists():
            # No audio produced — coverage check (a different cycle) owns this.
            continue

        try:
            head_wer, tail_wer = _window_wers(
                client, episode_path, script, window_s, language
            )
        except STTServiceError as exc:
            logger.warning(
                "Fidelity check skipped for %s: STT unavailable (%s)",
                chapter_id,
                exc,
            )
            continue

        ratio = _duration_ratio(episode_path, script)
        episode_wer = max(head_wer, tail_wer)
        suspect = (
            head_wer > threshold
            or tail_wer > threshold
            or ratio < ratio_threshold
        )

        # Track the lowest-WER attempt so we can restore it on exhaustion.
        if best_wer.get(chapter_id) is None or episode_wer < best_wer[chapter_id]:
            best_wer[chapter_id] = episode_wer
            _save_best_candidate(creator, episode_number, episode_path)

        budget = retry_budget.setdefault(chapter_id, {})
        retried_before = _CYCLE in budget

        if not suspect:
            # Episode passes. If it had been retried, record a resolved flag once.
            if retried_before and not _has_cycle_flag(flags, chapter_id):
                _append_flag(
                    flags,
                    chapter_id,
                    reason=(
                        f"WER {episode_wer:.2f} within {threshold:.2f} "
                        "after re-chunk"
                    ),
                    exhausted=False,
                )
            continue

        remaining = budget.get(_CYCLE, MAX_BUDGET_PER_CYCLE)
        reason = _suspect_reason(head_wer, tail_wer, ratio, threshold, ratio_threshold)
        if remaining > 0:
            budget[_CYCLE] = remaining - 1
            pending_retry.append(episode_number)
            logger.info(
                "Fidelity flagged %s (%s); scheduling re-chunk retry "
                "(%d attempt(s) left).",
                chapter_id,
                reason,
                remaining - 1,
            )
        else:
            # Budget exhausted: keep the best attempt, write an exhausted flag.
            _restore_best_candidate(creator, episode_number, episode_path)
            _append_flag(
                flags,
                chapter_id,
                reason=(
                    f"WER {best_wer[chapter_id]:.2f} exceeds {threshold:.2f} "
                    f"after {MAX_BUDGET_PER_CYCLE} attempts"
                ),
                exhausted=True,
            )
            logger.warning(
                "Fidelity check exhausted for %s; keeping best-effort "
                "artifact (WER %.2f).",
                chapter_id,
                best_wer[chapter_id],
            )

    return {
        "retry_budget": retry_budget,
        "best_wer": best_wer,
        "flags": flags,
        "pending_retry": pending_retry,
    }


def fidelity_route(state: GraphState) -> str:
    """Conditional edge: loop back to ``synthesize`` while retries are pending."""
    return "synthesize" if state.get("pending_retry") else "assemble"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_stt_client(creator: Any) -> Optional[STTClient]:
    """Return the creator's injected STT client, or build one from env.

    Returns ``None`` only when client construction itself fails — a reachable
    service is not verified here (per-call errors degrade gracefully instead).
    """
    injected = getattr(creator, "stt_client", None)
    if injected is not None:
        return injected  # type: ignore[return-value]
    try:
        base_url = os.getenv("STT_BASE_URL", "http://localhost:8888")
        return WhisperSTTClient(base_url=base_url)
    except Exception as exc:  # pragma: no cover - construction is trivial
        logger.warning("Could not build STT client: %s", exc)
        return None


def _episode_path(creator: Any, episode_number: int) -> Path:
    return Path(creator.episodes_path) / f"episode_{episode_number:03d}.mp3"


def _best_candidate_path(creator: Any, episode_number: int) -> Path:
    # Stored under a subdirectory so it is never picked up by the
    # ``episode_*.mp3`` glob that create_m4b uses to assemble the book.
    cache = Path(creator.episodes_path) / ".fidelity_best"
    return cache / f"episode_{episode_number:03d}.mp3"


def _save_best_candidate(creator: Any, episode_number: int, source: Path) -> None:
    dest = _best_candidate_path(creator, episode_number)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    except OSError as exc:
        logger.warning("Could not cache best fidelity candidate: %s", exc)


def _restore_best_candidate(
    creator: Any, episode_number: int, dest: Path
) -> None:
    src = _best_candidate_path(creator, episode_number)
    if not src.exists():
        return
    try:
        shutil.copy2(src, dest)
    except OSError as exc:
        logger.warning("Could not restore best fidelity candidate: %s", exc)


def _window_wers(
    client: STTClient,
    episode_path: Path,
    script: str,
    window_s: float,
    language: Optional[str],
) -> tuple[float, float]:
    """Return ``(head_wer, tail_wer)`` for the episode's boundary windows."""
    from audify.utils.audio import AudioProcessor

    duration = AudioProcessor.get_duration(episode_path)
    if duration <= 0:
        # Undecodable / empty audio is a total failure of both windows.
        return 1.0, 1.0

    window = min(window_s, duration)

    head_hyp = client.transcribe(
        episode_path, start_s=0.0, end_s=window, language=language
    )
    tail_hyp = client.transcribe(
        episode_path,
        start_s=max(duration - window_s, 0.0),
        end_s=duration,
        language=language,
    )

    head_wer = word_error_rate(
        comparable_reference(script, head_hyp, side="head"), head_hyp
    )
    tail_wer = word_error_rate(
        comparable_reference(script, tail_hyp, side="tail"), tail_hyp
    )
    return head_wer, tail_wer


def _duration_ratio(episode_path: Path, script: str) -> float:
    """Actual audio seconds / expected seconds from script word count."""
    from audify.utils.audio import AudioProcessor

    word_count = len(script.split())
    expected_minutes = max(word_count / WORDS_PER_MINUTE, 1)
    expected_s = expected_minutes * 60
    actual_s = AudioProcessor.get_duration(episode_path)
    return actual_s / expected_s if expected_s else 0.0


def _suspect_reason(
    head_wer: float,
    tail_wer: float,
    ratio: float,
    threshold: float,
    ratio_threshold: float,
) -> str:
    parts = []
    if head_wer > threshold:
        parts.append(f"head WER {head_wer:.2f}")
    if tail_wer > threshold:
        parts.append(f"tail WER {tail_wer:.2f}")
    if ratio < ratio_threshold:
        parts.append(f"duration ratio {ratio:.2f}")
    return ", ".join(parts) or "below thresholds"


def _has_cycle_flag(flags: dict[str, list[FlagEntry]], chapter_id: str) -> bool:
    return any(
        entry["cycle"] == _CYCLE for entry in flags.get(chapter_id, [])
    )


def _append_flag(
    flags: dict[str, list[FlagEntry]],
    chapter_id: str,
    *,
    reason: str,
    exhausted: bool,
) -> None:
    entry: FlagEntry = {
        "cycle": "cycle_3_retry",
        "reason": reason,
        "exhausted": exhausted,
    }
    flags.setdefault(chapter_id, []).append(entry)
