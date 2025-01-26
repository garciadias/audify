# %%
from pathlib import Path

from audify import ebook_read, text_to_speech

MODULE_PATH = Path(__file__).resolve().parents[1]


# %%
if __name__ == "__main__":
    # %%
    text = ebook_read.read_chapters(f"{MODULE_PATH}/data/test.epub")
    with open(f"{MODULE_PATH}/data/output/chapters.txt", "w") as f:
        for i, chapter in enumerate(text, start=1):
            print(f"Synthesizing chapter: {i}")
            text_to_speech.synthesize_chapter(chapter, i)
            title = ebook_read.get_chapter_title(chapter)
            f.write(f"Chapter {i}: {title}\n")
    print("All chapters synthesized.")
    # %%
    book_title = "dust"
    # Create M4B file
    chapter_files = Path(f"{MODULE_PATH}/data/output/{book_title}").rglob("*.wav")
    text_to_speech.create_m4b(
        chapter_files=chapter_files,
        filename=f"{MODULE_PATH}/data/test.epub",
        cover_image_path=f"{MODULE_PATH}/data/output/dust/cover.jpg"
    )
    print("M4B file created.")

# %%
