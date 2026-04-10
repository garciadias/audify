#!/usr/bin/env python3
"""Debug script to examine EPUB structure."""

import sys
import os

sys.path.insert(0, ".")

from ebooklib import epub, ITEM_DOCUMENT, ITEM_NAVIGATION, ITEM_COVER, ITEM_IMAGE
import bs4


def examine_epub(epub_path):
    print(f"Examining EPUB: {epub_path}")
    book = epub.read_epub(epub_path)

    print("\n=== ITEMS IN EPUB ===")
    for i, item in enumerate(book.get_items()):
        item_type = item.get_type()
        item_name = item.get_name()
        print(f"{i:3d}: {item_type:20} {item_name}")

        if item_type == ITEM_DOCUMENT:
            # Try to get a snippet of content
            try:
                content = item.get_body_content().decode("utf-8", errors="ignore")
                soup = bs4.BeautifulSoup(content, "html.parser")
                # Extract first 200 chars of text
                text = soup.get_text()[:200].replace("\n", " ").strip()
                print(f"     Preview: {text}")

                # Check if it's a navigation/toc item
                if "toc" in item_name.lower() or "nav" in item_name.lower():
                    print("     *** This appears to be a TOC/navigation file")
                # Check for common TOC patterns in content
                soup_lower = str(soup).lower()
                if (
                    "table of contents" in soup_lower
                    or "contents" in soup_lower
                    and "chapter" in soup_lower
                ):
                    print("     *** Contains TOC-like text")

            except Exception as e:
                print(f"     Error reading: {e}")

    print("\n=== DOCUMENT ITEMS ONLY ===")
    doc_items = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]
    for i, item in enumerate(doc_items):
        print(f"{i:3d}: {item.get_name()}")

    print("\n=== TOC/NAVIGATION ITEMS ===")
    nav_items = [
        item for item in book.get_items() if item.get_type() == ITEM_NAVIGATION
    ]
    for i, item in enumerate(nav_items):
        print(f"{i:3d}: {item.get_name()}")

    # Check book TOC
    print("\n=== BOOK TOC (ebooklib) ===")
    toc = book.toc
    if toc:
        for entry in toc:
            print(f"TOC entry: {entry}")
    else:
        print("No TOC found in book.toc")

    # Check spine (reading order)
    print("\n=== SPINE (reading order) ===")
    for i, (spine_id, _) in enumerate(book.spine):
        item = book.get_item_with_id(spine_id)
        if item:
            print(f"{i:3d}: {item.get_name()} ({spine_id})")
        else:
            print(f"{i:3d}: ID '{spine_id}' not found")

    return book


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_epub.py <epub_file>")
        sys.exit(1)

    epub_path = sys.argv[1]
    if not os.path.exists(epub_path):
        print(f"File not found: {epub_path}")
        sys.exit(1)

    examine_epub(epub_path)
