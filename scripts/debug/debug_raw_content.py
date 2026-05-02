import sys

from audify.readers.ebook import EpubReader


def debug_raw(file_path):
    reader = EpubReader(file_path)
    for i, (spine_id, _) in enumerate(reader.book.spine):
        item = reader.book.get_item_with_id(spine_id)
        if item and item.get_type() == 9:  # ITEM_DOCUMENT
            print(f"--- Item {i}: {item.get_name()} ---")
            content = item.get_body_content().decode("utf-8", errors="ignore")
            print(content[:500])
            print("-" * 30)
        if i > 10:
            break


if __name__ == "__main__":
    debug_raw(sys.argv[1])
