from __future__ import annotations

import logging
from pathlib import Path

from audify.qa.state import GraphState
from audify.text_to_speech import TTSSynthesisError

logger = logging.getLogger(__name__)


def synthesize_node(state: GraphState) -> dict:
    """Synthesize TTS audio for every script (Phase 3 of the legacy pipeline)."""
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

    return {"episode_paths": episode_paths}
