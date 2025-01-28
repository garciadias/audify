# %%
from pathlib import Path

from ebooklib import epub

from audify import ebook_read, text_to_speech

MODULE_PATH = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    # %%
    # Read the EPUB file and extract Metadata
    book = epub.read_epub(f"{MODULE_PATH}/data/test.epub")
    book_title = ebook_read.get_book_title(book)
    # Prepare the output directory
    audio_book_path = Path(f"{MODULE_PATH}/data/output/{book_title}")
    audio_book_path.mkdir(parents=True, exist_ok=True)
    cover_path = ebook_read.save_book_cover_image(book)
    with open(audio_book_path / "chapters.txt", "w") as f:
        f.write(";FFMETADATA1\n")
        f.write("major_brand=M4A\n")
        f.write("minor_version=512\n")
        f.write("compatible_brands=M4A isomiso2\n")
        f.write("encoder=Lavf61.7.100\n")

    # Process chapters
    text_to_speech.process_chapters(book, audio_book_path)
    # Combine chapters into a single M4B file
    chapter_files = Path(audio_book_path).rglob("*.wav")
    text_to_speech.create_m4b(
        chapter_files=chapter_files,
        filename=f"{audio_book_path}/../{book_title}.epub",
        cover_image_path=cover_path,
    )
