from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from audify.qa.nodes.report import MAX_BUDGET_PER_CYCLE
from audify.qa.state import CycleId, GraphState
from audify.text_to_speech import TTSSynthesisError

logger = logging.getLogger(__name__)

# Floor for the re-chunk batch size so shrinking never collapses to zero.
MIN_BATCH_CHARS = 200

_CYCLE: CycleId = "cycle_3_retry"


def synthesize_node(state: GraphState) -> dict:
    """Synthesize TTS audio for every script (Phase 3 of the legacy pipeline).

    Two modes of operation:

    * **First pass** (``pending_retry`` empty): synthesize every script and mark
      all episodes for the fidelity check.
    * **Retry pass** (``pending_retry`` populated by the cycle-3 fidelity edge):
      re-synthesize *only* the flagged episodes with a smaller TTS batch size
      (re-chunk), and mark just those episodes for re-evaluation.
    """
    pending_retry = state.get("pending_retry", [])
    if pending_retry:
        return _resynthesize(state, pending_retry)

    creator = state["creator"]
    chapter_scripts = state["chapter_scripts"]
    chapter_titles = state["chapter_titles"]

    episode_paths: list[Path] = []

    creator.progress.set_phase("Synthesizing")

    for episode_number, audiobook_script in chapter_scripts:
        chapter_title = chapter_titles[episode_number - 1]
        text_snippet = " ".join(audiobook_script.split()[:100])

        try:
            creator.progress.print_chapter_start(
                episode_number, chapter_title, text_snippet
            )
            episode_path = creator.synthesize_episode(audiobook_script, episode_number)

            if episode_path.exists():
                episode_paths.append(episode_path)
                logger.info(
                    f"Successfully created Episode {episode_number}: {episode_path}"
                )
            else:
                logger.warning(f"Failed to create Episode {episode_number}")

        except TTSSynthesisError:
            raise
        except Exception as e:
            logger.error(
                f"Error creating Episode {episode_number}: {e}",
                exc_info=True,
            )
            raise

    return {
        "episode_paths": episode_paths,
        "episodes_to_check": [n for n, _ in chapter_scripts],
    }


def _resynthesize(state: GraphState, pending_retry: list[int]) -> dict:
    """Re-synthesize the flagged episodes with a smaller batch size.

    Episode file paths are deterministic (``episode_NNN.mp3``), so the global
    ``episode_paths`` list is unchanged; only the audio behind those paths is
    regenerated. Returns the re-synthesized episode numbers as the next
    ``episodes_to_check`` and clears ``pending_retry``.
    """
    creator = state["creator"]
    scripts = dict(state["chapter_scripts"])
    retry_budget = state.get("retry_budget", {})
    base = _tts_base_length(creator)

    creator.progress.set_phase("Synthesizing")

    for episode_number in pending_retry:
        script = scripts.get(episode_number)
        if script is None:
            continue

        chapter_id = f"chapter_{episode_number}"
        remaining = retry_budget.get(chapter_id, {}).get(
            _CYCLE, MAX_BUDGET_PER_CYCLE
        )
        # Attempts already spent → shrink the batch geometrically each retry.
        attempt = MAX_BUDGET_PER_CYCLE - remaining
        max_text_length = (
            max(base // (attempt + 1), MIN_BATCH_CHARS) if base else None
        )

        _remove_stale_audio(creator, episode_number)

        try:
            episode_path = creator.synthesize_episode(
                script, episode_number, max_text_length=max_text_length
            )
        except TTSSynthesisError:
            raise
        except Exception as e:
            logger.error(
                f"Error re-synthesizing Episode {episode_number}: {e}",
                exc_info=True,
            )
            raise

        if episode_path.exists():
            logger.info(
                f"Re-synthesized Episode {episode_number} "
                f"(batch≤{max_text_length} chars): {episode_path}"
            )
        else:
            logger.warning(f"Re-synthesis produced no audio for {episode_number}")

    return {"episodes_to_check": list(pending_retry), "pending_retry": []}


def _tts_base_length(creator: Any) -> Optional[int]:
    """Best-effort read of the provider's default batch size (chars)."""
    try:
        return int(creator._get_tts_config().max_text_length)
    except Exception:
        return None


def _remove_stale_audio(creator: Any, episode_number: int) -> None:
    """Delete the stale episode artifacts so synthesize_episode regenerates.

    ``synthesize_episode`` early-returns when the MP3 already exists, so the
    retry must clear it (and the intermediate WAV) first.
    """
    episodes_path = Path(creator.episodes_path)
    stem = f"episode_{episode_number:03d}"
    for suffix in (".mp3", ".wav"):
        path = episodes_path / f"{stem}{suffix}"
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not remove stale %s: %s", path, exc)
