# %%
import bs4
from ebooklib import ITEM_DOCUMENT, epub


def read_chapters(path: str) -> list[str]:
    book = epub.read_epub(path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            chapters.append(item.get_body_content())
    return chapters


def extract_text_from_epub_chapter(chapter: str) -> str:
    soup = bs4.BeautifulSoup(chapter, "html.parser")
    return soup.get_text()


def break_text_into_sentences(text: str) -> list[str]:
    return text.split(".")
