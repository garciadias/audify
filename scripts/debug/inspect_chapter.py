import sys
from ebooklib import epub

def inspect_chapter(file_path, chapter_id):
    book = epub.read_epub(file_path)
    item = book.get_item_with_id(chapter_id)
    if item:
        print(f"--- {item.get_name()} ---")
        print(item.get_body_content().decode("utf-8", errors="ignore")[:1000])
    else:
        print("Item not found")

if __name__ == "__main__":
    inspect_chapter(sys.argv[1], sys.argv[2])
