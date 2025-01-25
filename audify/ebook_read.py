# %%
from ebooklib import ITEM_DOCUMENT, epub


def read_chapters(path: str) -> list[str]:
    book = epub.read_epub(path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            chapters.append(item.get_body_content())
    return chapters
