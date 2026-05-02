import sys

import bs4

from audify.readers.ebook import EpubReader


def debug_toc(file_path):
    reader = EpubReader(file_path)
    toc_names = reader._build_toc_item_name_set()

    current_group = []
    for spine_id, _ in reader.book.spine:
        item = reader.book.get_item_with_id(spine_id)
        if not item or item.get_type() != 9:
            continue
        item_name = item.get_name().lower()
        if item_name in toc_names:
            if current_group:
                merged = reader._merge_items(current_group)
                if merged:
                    soup = bs4.BeautifulSoup(merged, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    links = soup.find_all("a")
                    list_items = soup.find_all(["li", "dt", "dd"])
                    print(f"--- Chapter: {reader.get_chapter_title(merged)} ---")
                    print(f"Links: {len(links)}, ListItems: {len(list_items)}")
                    print(f"Text snippet: {text[:300]}...")
                    if reader._looks_like_toc(soup, text):
                        print("Result: Looks like TOC")
                    else:
                        print("Result: Valid")
                    print("-" * 40)
            current_group = [item]
        else:
            current_group.append(item)


if __name__ == "__main__":
    debug_toc(sys.argv[1])
