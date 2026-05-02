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
        if not item or item.get_type() != 9: # ITEM_DOCUMENT
            continue
        item_name = item.get_name().lower()
        if any(token in item_name for token in ["toc", "nav", "titlepage", "cover", "copyright"]):
            continue
        
        if item_name in toc_names:
            matches_found += 1
            if current_group:
                merged = reader._merge_items(current_group)
                valid = reader._is_valid_chapter(merged)
                print(f"Checking group ending before {item_name}: Valid={valid}")
                if not valid:
                    # We need to know WHY it's invalid. 
                    # Let's override _is_valid_chapter for a moment or just replicate logic.
                    text = reader.extract_text(merged).lower()
                    soup = reader.bs4.BeautifulSoup(merged, "html.parser") # Wait reader.bs4 doesn't exist, it's just bs4
                    # I'll just use the actual reader's methods if I can.
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
    import bs4 # needed for the logic inside if I were to replicate it
    debug_validity(sys.argv[1])
