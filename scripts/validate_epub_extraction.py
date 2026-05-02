import sys
from pathlib import Path
from audify.readers.ebook import EpubReader

def test_epub(file_path):
    print(f"Testing: {file_path}")
    try:
        reader = EpubReader(file_path)
        chapters = reader.get_chapters()
        print(f"Found {len(chapters)} chapters.")
        for i, chapter in enumerate(chapters):
            title = reader.get_chapter_title(chapter)
            text = reader.extract_text(chapter)
            print(f"Chapter {i+1}: {title} (Length: {len(text)} chars)")
            # Print first 100 chars of text to verify
            snippet = text[:100].replace('\n', ' ')
            print(f"Snippet: {snippet}...")
            print("-" * 20)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    ebooks_dir = Path("./data/ebooks")
    for epub_file in ebooks_dir.glob("*.epub"):
        test_epub(epub_file)
