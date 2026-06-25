from __future__ import annotations

import logging

from audify.audiobook_creator import clean_text_for_audiobook
from audify.qa.state import GraphState

logger = logging.getLogger(__name__)


def script_gen_node(state: GraphState) -> dict:
    """Generate LLM scripts for every chapter (Phase 1 of the legacy pipeline).

    NOTE: this loop intentionally duplicates the script-generation block in
    ``AudiobookCreator.create_audiobook_series`` (audiobook_creator.py:953-1001).
    The legacy path stays the default while the graph evolves; once cyclic
    detectors land and ``--graph`` becomes default, the legacy paths get
    deleted — extracting a shared helper now would create churn for code that
    is about to disappear.
    """
    creator = state["creator"]
    chapters = state["chapters"]
    chapter_titles = state["chapter_titles"]
    total_chapters = len(chapters)

    script_word_counts: list[tuple[str, int]] = []
    chapter_scripts: list[tuple[int, str]] = []

    creator.progress.set_phase("Generating")

    for i, chapter_content in enumerate(chapters):
        episode_number = i + 1
        chapter_title = chapter_titles[i]

        creator.progress.set_counter(i + 1, total_chapters)

        cleaned_content = clean_text_for_audiobook(chapter_content)
        text_snippet = " ".join(cleaned_content.split()[:100])

        try:
            creator.progress.print_chapter_start(
                episode_number, chapter_title, text_snippet
            )
            audiobook_script = creator.generate_audiobook_script(
                chapter_content, episode_number
            )
            chapter_scripts.append((episode_number, audiobook_script))

            if audiobook_script:
                word_count = len(audiobook_script.split())
                script_word_counts.append((chapter_title, word_count))

        except Exception as e:
            logger.error(
                f"Error generating script for Episode {episode_number}: {e}",
                exc_info=True,
            )
            if len(creator.chapter_titles) < episode_number:
                creator.chapter_titles.append(chapter_title)

    creator.progress.set_phase("Validating")
    creator._validate_chapters(script_word_counts)
    creator._save_chapter_titles()

    return {"chapter_scripts": chapter_scripts}
