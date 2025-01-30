# %%
import re
from dataclasses import dataclass
from pathlib import Path

import bs4
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE, epub

MODULE_PATH = Path(__file__).resolve().parents[1]


@dataclass
class Chapter:
    title: str | None
    sentences: list[str]


class BookReader:
    def __init__(self, book: epub.EpubBook):
        self.book = book

    def read_chapters(self) -> list[Chapter]:
        chapters = []
        for item in self.book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                raw_content = item.get_body_content()
                chapter = self.extract_text_from_epub_chapter(raw_content)
                sentences = self.break_text_into_sentences(chapter)
                title = self.get_chapter_title(chapter)
                chapters.append(Chapter(title, sentences))
        return chapters

    def extract_text_from_epub_chapter(self, chapter: str) -> str:
        return bs4.BeautifulSoup(chapter, "html.parser").get_text()

    def break_text_into_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?;:¿¡]) +", text)
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            while len(sentence) > 239:
                result.append(sentence[:239])
                sentence = sentence[239:]
            if sentence:
                result.append(sentence)
        return result

    def get_book_title(self) -> str:
        title = self.book.get_metadata("DC", "title")[0][0]
        title = re.sub(r"(?<!^)(?=[A-Z])", "_", title).lower()
        return title

    def save_book_cover_image(self) -> str | None:
        # If ITEM_COVER is available, use it
        cover_image = next(
            (item for item in self.book.get_items() if item.get_type() == ITEM_COVER),
            None,
        )
        if not cover_image:
            # If not, use the first image
            cover_image = next(
                (
                    item
                    for item in self.book.get_items()
                    if item.get_type() == ITEM_IMAGE
                ),
                None,
            )
        if not cover_image:
            return None
        title = self.get_book_title()
        cover_path = f"{MODULE_PATH}/data/output/{title}/cover.jpg"
        with open(cover_path, "wb") as f:
            f.write(cover_image.content)
        return cover_path

    def get_language(self) -> str:
        language = self.book.get_metadata("DC", "language")[0][0]
        return language

    def get_chapter_title(self, chapter: str) -> str:
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
            return h1_tag.get_text()
        else:
            return "Unknown"
