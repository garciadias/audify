import sys

from audify.readers.ebook import EpubReader


def debug_toc(file_path):
    print(f"Debugging TOC for: {file_path}")
    reader = EpubReader(file_path)
    toc_names = reader._build_toc_item_name_set()
    print(f"TOC Item Names: {toc_names}")

    print("\nSpine Item Names:")
    for spine_id, _ in reader.book.spine:
        item = reader.book.get_item_with_id(spine_id)
        if item:
            print(f"- {item.get_name()}")


if __name__ == "__main__":
    debug_toc(sys.argv[1])
