# %%
from pathlib import Path

import bs4
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE, epub

from audify.domain.interface import Reader

MODULE_PATH = Path(__file__).resolve().parents[1]


class EpubReader(Reader):
    def __init__(self, path: str | Path):
        self.book = epub.read_epub(path)
        self.title = self.get_title()

    def get_chapters(self) -> list[str]:
        chapters = []
        for item in self.book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                chapters.append(item.get_body_content())
        return chapters

    def extract_text(self, chapter: str) -> str:
        return bs4.BeautifulSoup(chapter, "html.parser").get_text()

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
            return h1_tag.text
        else:
            return "Unknown"

    def get_title(self) -> str:
        title = self.book.title
        if not title and self.book.get_metadata("DC", "title"):
            if self.book.get_metadata("DC", "title")[0]:
                title = self.book.get_metadata("DC", "title")[0][0]
        if not title:
            "missing title"
        return title

    def get_cover_image(self, output_path: str | Path) -> Path | None:
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
        cover_path = f"{output_path}/cover.jpg"
        with open(cover_path, "wb") as f:
            f.write(cover_image.content)
        return Path(cover_path)

    def get_language(self) -> str:
        language = self.book.get_metadata("DC", "language")[0][0]
        return language
