import sys

import bs4

from audify.readers.ebook import EpubReader


def debug_validity_detailed(file_path):
    reader = EpubReader(file_path)
    toc_names = reader._build_toc_item_name_set()

    current_group = []

    for spine_id, _ in reader.book.spine:
        item = reader.book.get_item_with_id(spine_id)
        if not item or item.get_type() != 9:  # ITEM_DOCUMENT
            continue
        item_name = item.get_name().lower()

        if item_name in toc_names:
            if current_group:
                merged = reader._merge_items(current_group)
                if merged is None:
                    print(
                        f"INVALID: None (content could not be merged "
                        f"from group {current_group})"
                    )
                    continue
                title = reader.get_chapter_title(merged)
                # Try to find why it might be invalid
                is_valid = reader._is_valid_chapter(merged)
                if not is_valid:
                    print(f"INVALID: {title} (from group {current_group})")
                    # Dig deeper
                    if len(merged.strip()) < 100:
                        print(f"  - Reason: Length < 100 ({len(merged.strip())})")
                    soup = bs4.BeautifulSoup(merged, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    if len(text) < 80:
                        print(f"  - Reason: Visible text < 80 ({len(text)})")
                    if reader._looks_like_toc(soup, text.lower()):
                        print("  - Reason: Looks like TOC")
                    if reader._looks_like_copyright(text.lower()):
                        print("  - Reason: Looks like Copyright")
                else:
                    print(f"VALID: {title}")
            current_group = [item]
        else:
            current_group.append(item)

    if current_group:
        merged = reader._merge_items(current_group)
        if merged is None:
            print(
                f"INVALID: None (content could not be merged "
                f"from final group {current_group})"
            )
        else:
            title = reader.get_chapter_title(merged)
            if not reader._is_valid_chapter(merged):
                print(f"INVALID: {title} (final group)")


if __name__ == "__main__":
    debug_validity_detailed(sys.argv[1])
