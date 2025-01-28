from pathlib import Path

from audify import text_to_speech
from audify.ebook_read import BookReader


class BookSynthesizer:
    def __init__(self, book_path: str):

        self.book = BookReader(book_path)
        self.book_title = self.book.get_book_title()
        self.audio_book_path = Path(
            f"{Path(book_path).parent}/output/{self.book_title}"
        )
        self.audio_book_path.mkdir(parents=True, exist_ok=True)
        self.cover_path = self.book.save_book_cover_image()
        self.language = self.book.get_language()
        self._initialize_metadata()

    def _initialize_metadata(self):
        with open(self.audio_book_path / "chapters.txt", "w") as f:
            f.write(";FFMETADATA1\n")
            f.write("major_brand=M4A\n")
            f.write("minor_version=512\n")
            f.write("compatible_brands=M4A isomiso2\n")
            f.write("encoder=Lavf61.7.100\n")

    def synthesize(self):
        self._process_chapters()
        self._create_m4b()

    def _process_chapters(self):
        chapter_start = 0
        chapter_id = 1
        chapters = self.book.read_chapters()
        for chapter in chapters:
            if len(chapter) < 1000:
                continue
            chapter_start = text_to_speech.process_chapter(
                chapter_id, chapter, chapter_start, self.audio_book_path, self.language
            )
            chapter_id += 1

    def _create_m4b(self):
        chapter_files = Path(self.audio_book_path).rglob("*.wav")
        text_to_speech.create_m4b(
            chapter_files=chapter_files,
            filename=f"{self.audio_book_path}/../{self.book_title}.epub",
            cover_image_path=self.cover_path,
        )
