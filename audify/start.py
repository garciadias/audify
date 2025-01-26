# %%
from pathlib import Path

from ebooklib import epub

from audify import ebook_read, text_to_speech

MODULE_PATH = Path(__file__).resolve().parents[1]


# %%
if __name__ == "__main__":
    # %%
    book = epub.read_epub(f"{MODULE_PATH}/data/test.epub")
    chapters = ebook_read.read_chapters(book)
    book_title = ebook_read.get_book_title(book)
    audio_book_path = Path(f"{MODULE_PATH}/data/output/{book_title}")
    audio_book_path.mkdir(parents=True, exist_ok=True)
    Path(audio_book_path).mkdir(parents=True, exist_ok=True)
    cover_path = ebook_read.save_book_cover_image(book)
    chapter_start = 0
    chapter_id = 1
    for chapter in chapters:
        if len(chapter) < 1000:
            continue
        chapter_start = text_to_speech.process_chapter(
            chapter_id, chapter, chapter_start
        )
        chapter_id += 1
    # Create M4B file
    chapter_files = Path(audio_book_path).rglob("*.wav")
    text_to_speech.create_m4b(
        chapter_files=chapter_files,
        filename=f"{audio_book_path}/../{book_title}.epub",
        cover_image_path=cover_path,
    )

# %%
