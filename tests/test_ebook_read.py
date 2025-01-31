from pathlib import Path

from audify.ebook_read import EpubReader

MODULE_PATH = Path(__file__).resolve().parents[1]


def test_epub_reader_init(tmp_path):
    # Use a dummy .epub file path if needed
    dummy_epub_path = tmp_path / "dummy.epub"
    dummy_epub_path.touch()
    reader = EpubReader(dummy_epub_path)
    assert reader is not None


def test_get_chapters_returns_list(tmp_path):
    dummy_epub_path = tmp_path / "dummy.epub"
    dummy_epub_path.touch()
    reader = EpubReader(dummy_epub_path)
    chapters = reader.get_chapters()
    assert isinstance(chapters, list)
