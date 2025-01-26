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


def extract_text_from_epub_chapter(chapter: str) -> tuple[str, str]:
    return bs4.BeautifulSoup(chapter, "html.parser").get_text()


def break_text_into_sentences(text: str) -> list[str]:
    return text.split(".")


def get_chapter_title(chapter: str) -> str:
    possible_titles = [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    ]
    soup = bs4.BeautifulSoup(chapter, "html.parser")
    h1_tag = soup.find(possible_titles)
    if h1_tag:
        return h1_tag.text
    else:
        return "Unknown"
