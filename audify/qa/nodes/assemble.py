from __future__ import annotations

import logging

from audify.qa.state import GraphState

logger = logging.getLogger(__name__)


def assemble_node(state: GraphState) -> dict:
    """Assemble episode MP3s into a final M4B file."""
    creator = state["creator"]
    episode_paths = state["episode_paths"]

    if episode_paths:
        creator.create_m4b()
        logger.info(f"Audiobook series complete with {len(episode_paths)} episodes.")
    else:
        logger.warning("No episode paths to assemble — skipping M4B creation.")

    return {}
