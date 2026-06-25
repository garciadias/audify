from __future__ import annotations

import logging

from audify.qa.state import GraphState

logger = logging.getLogger(__name__)


def load_scripts_node(state: GraphState) -> dict:
    """Load previously-saved scripts from disk for ``mode == "synthesize"``.

    Mirrors the setup phase of
    :meth:`AudiobookCreator._synthesize_from_existing_scripts`: scans
    ``scripts_path/episode_*_script.txt`` and ``chapter_titles.json``,
    populating ``chapter_scripts`` and ``chapter_titles`` so that
    ``synthesize_node`` can run unchanged.
    """
    creator = state["creator"]

    creator.progress.set_phase("Loading scripts")

    script_files = sorted(creator.scripts_path.glob("episode_*_script.txt"))
    if not script_files:
        logger.error(
            "No existing scripts found for synthesize-only mode. "
            "Run with --process-only (or no flag) first to generate scripts."
        )
        return {
            "chapter_scripts": [],
            "chapter_titles": [],
        }

    chapter_titles = creator._load_chapter_titles()
    chapter_scripts: list[tuple[int, str]] = []

    for script_file in script_files:
        try:
            episode_num = int(script_file.stem.split("_")[1])
        except (IndexError, ValueError):
            logger.warning(
                f"Could not parse episode number from {script_file.name}"
            )
            continue

        try:
            audiobook_script = script_file.read_text(encoding="utf-8")
        except IOError as e:
            logger.warning(f"Could not read {script_file}: {e}")
            continue

        chapter_scripts.append((episode_num, audiobook_script))

    creator.chapter_titles = chapter_titles
    return {
        "chapter_scripts": chapter_scripts,
        "chapter_titles": chapter_titles,
    }
