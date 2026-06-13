from __future__ import annotations

import logging

from audify.qa.state import GraphState

logger = logging.getLogger(__name__)


def confirm_node(state: GraphState) -> dict:
    """Display the table of contents and (optionally) prompt the user.

    Runs between ``read`` and ``script_gen`` for the ``"full"`` and
    ``"process"`` graphs. Mirrors the legacy TOC display + confirmation
    prompt in ``AudiobookCreator.create_audiobook_series``.

    Abort semantics: if the user declines, the node returns
    ``{"chapters": []}``. Downstream nodes already handle an empty
    chapter list as a no-op, so the abort propagates without conditional
    edges.
    """
    creator = state["creator"]
    chapter_titles = state["chapter_titles"]
    chapters = state["chapters"]
    num_chapters = len(chapters)

    creator.progress.stop()
    creator.progress.print_table_of_contents(chapter_titles)
    creator.progress.start()

    if creator.confirm:
        creator.progress.stop()
        try:
            response = input(
                f"Create {num_chapters} audiobook episodes? (y/N): "
            )
        except (EOFError, OSError):
            # Non-tty (CI / pipe / captured stdin): treat as no-input
            # so the run aborts cleanly rather than hanging or raising.
            response = ""
        if response.lower() not in ("y", "yes"):
            logger.info("Audiobook creation cancelled by user.")
            return {"chapters": []}
        creator.progress.start()

    return {}
