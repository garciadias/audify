from pathlib import Path

import bs4
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE, epub
from typing_extensions import Reader

MODULE_PATH = Path(__file__).resolve().parents[1]


class EpubReader(Reader):
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.book = self.read()
        self.title = self.get_title()

    def read(self):
        return epub.read_epub(self.path)

    def get_chapters(self) -> list[str]:
        chapters = []
        for item in self.book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                chapters.append(item.get_body_content().decode("utf-8"))
        return chapters

    def extract_text(self, chapter: str) -> str:
        return bs4.BeautifulSoup(chapter, "html.parser").get_text()

    def get_chapter_title(self, chapter: str) -> str:
        possible_titles = [f"h{i}" for i in range(1, 7)]
        possible_titles += ["title", "hgroup", "header"]
        soup = bs4.BeautifulSoup(chapter, "html.parser")
        title_tag = soup.find(possible_titles)
        if title_tag:
            return title_tag.text
        else:
            return "Unknown"

    def get_title(self) -> str:
        title = self.book.title
        if not title and self.book.get_metadata("DC", "title"):
            if self.book.get_metadata("DC", "title")[0]:
                title = self.book.get_metadata("DC", "title")[0][0]
        if not title:
            title = "missing title"
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
