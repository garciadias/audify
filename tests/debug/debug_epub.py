import sys
from pathlib import Path
from ebooklib import epub

def debug_epub(file_path):
    print(f"Debugging: {file_path}")
    book = epub.read_epub(file_path)
    
    print("\n--- Spine ---")
    for spine_id, _ in book.spine:
        item = book.get_item_with_id(spine_id)
        if item:
            print(f"ID: {spine_id} | Name: {item.get_name()} | Type: {item.get_type()}")

    print("\n--- TOC ---")
    print(book.toc)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_epub(sys.argv[1])
    else:
        print("Please provide a file path")
