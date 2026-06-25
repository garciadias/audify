from __future__ import annotations

from audify.qa.state import GraphState
from audify.readers.ebook import EpubReader


def read_node(state: GraphState) -> dict:
    """Extract chapters and titles from the source file via the existing reader."""
    creator = state["creator"]

    creator.progress.set_phase("Reading")

    if isinstance(creator.reader, EpubReader):
        chapters = creator.reader.get_chapters()
    else:
        chapters = [creator.reader.cleaned_text]

    if creator.max_chapters:
        chapters = chapters[: creator.max_chapters]

    chapter_titles = []
    for i, chapter_content in enumerate(chapters, 1):
        if isinstance(creator.reader, EpubReader):
            title = str(creator.reader.get_chapter_title(chapter_content))
        else:
            title = f"Chapter {i}"
        chapter_titles.append(title)

    return {
        "chapters": chapters,
        "chapter_titles": chapter_titles,
        "retry_budget": {},
        "best_wer": {},
        "flags": {},
    }
