from pathlib import Path

from audify import ebook_read

MODULE_PATH = Path(__file__).resolve().parents[1]


def test_read_chapters():
    text = ebook_read.get_chapters(f"{MODULE_PATH}/data/test.epub")
    assert len(text) == 49
    assert len([len(t) for t in text if len(t) > 1000]) == 42


def test_extract_text_from_epub_chapter():
    text = ebook_read.get_chapters(f"{MODULE_PATH}/data/test.epub")
    text = ebook_read.extract_text(text[10])
    print(text)
    assert len(text) > 1000
    # verify if html tags are removed
    assert "<" not in text
