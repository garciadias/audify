import sys

from audify.readers.ebook import EpubReader


def debug_validity(file_path):
    reader = EpubReader(file_path)
    toc_names = reader._build_toc_item_name_set()

    chapters = []
    current_group = []
    matches_found = 0

    for spine_id, _ in reader.book.spine:
        item = reader.book.get_item_with_id(spine_id)
        if not item or item.get_type() != 9:  # ITEM_DOCUMENT
            continue
        item_name = item.get_name().lower()
        if any(
            token in item_name
            for token in ["toc", "nav", "titlepage", "cover", "copyright"]
        ):
            continue

        if item_name in toc_names:
            matches_found += 1
            if current_group:
                merged = reader._merge_items(current_group)
                valid = reader._is_valid_chapter(merged)
                print(f"Checking group ending before {item_name}: Valid={valid}")
                if not valid:
                    # Note: reader.bs4 is not available.
                    pass
                if valid:
                    chapters.append(merged)
            current_group = [item]
        else:
            current_group.append(item)

    if current_group:
        merged = reader._merge_items(current_group)
        valid = reader._is_valid_chapter(merged)
        print(f"Checking final group: Valid={valid}")


if __name__ == "__main__":
    debug_validity(sys.argv[1])
