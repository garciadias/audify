from pathlib import Path

from audify import ebook_read

MODULE_PATH = Path(__file__).resolve().parents[1]


def test_read_chapters():
    text = ebook_read.read_chapters(f"{MODULE_PATH}/data/test.epub")
    assert len(text) == 49
    assert len([len(t) for t in text if len(t) > 1000]) == 42
