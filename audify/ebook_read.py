# %%
import re
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

    def break_text_into_sentences(self, text: str) -> list[str]:
        # Replace @ with 'a' to avoid TTS errors
        text = text.replace("@", "a")
        sentences = re.split(r"(?<=[.!?;:¿¡]) +", text)
        # Remove extra spaces
        sentences = [re.sub(r" +", " ", sentence) for sentence in sentences]
        # Remove leading and trailing spaces
        sentences = [sentence.strip() for sentence in sentences]
        # Remove empty sentences
        sentences = [sentence for sentence in sentences if sentence]
        # Split long sentences into smaller ones to avoid TTS errors
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            while len(sentence) > 239:
                result.append(sentence[:239])
                sentence = sentence[239:]
            if sentence:
                result.append(sentence)
        return result

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
        title = self.book.title or self.book.get_metadata("DC", "title")[0][0]
        return title

    def get_cover_image(self) -> str | None:
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
        title = self.get_file_name_title()
        cover_path = f"{MODULE_PATH}/data/output/{title}/cover.jpg"
        with open(cover_path, "wb") as f:
            f.write(cover_image.content)
        return cover_path

    def get_language(self) -> str:
        language = self.book.get_metadata("DC", "language")[0][0]
        return language

    def get_file_name_title(self) -> str:
        # Make title snake_case and remove special characters and spaces
        title = self.get_title().lower().replace(" ", "_")
        # replace multiple underscores with a single one
        title = re.sub(r"_+", "_", title)
        # Remove leading and trailing underscores
        title = title.strip("_")
        # Remove letter accents
        return title
