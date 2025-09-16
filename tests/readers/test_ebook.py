import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE

from audify.readers.ebook import EpubReader


@pytest.fixture
def temp_epub_path():
    """Create a temporary EPUB file path."""
    with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_epub_book():
    """Create a mock EPUB book object."""
    mock_book = Mock()
    mock_book.title = "Test Book Title"
    mock_book.get_metadata.return_value = [("Test Metadata Title", {})]
    return mock_book


@pytest.fixture
def mock_epub_items():
    """Create mock EPUB items (chapters, images, covers)."""
    # Mock document item (chapter)
    mock_chapter1 = Mock()
    mock_chapter1.get_type.return_value = ITEM_DOCUMENT
    mock_chapter1.get_body_content.return_value = b'<html><body><h1>Chapter 1</h1><p>' \
        b'This is chapter 1 content.</p></body></html>'

    mock_chapter2 = Mock()
    mock_chapter2.get_type.return_value = ITEM_DOCUMENT
    mock_chapter2.get_body_content.return_value = b'<html><body><h2>Chapter 2</h2><p>' \
        b'This is chapter 2 content.</p></body></html>'

    # Mock cover item
    mock_cover = Mock()
    mock_cover.get_type.return_value = ITEM_COVER
    mock_cover.content = b'fake_cover_image_data'

    # Mock image item
    mock_image = Mock()
    mock_image.get_type.return_value = ITEM_IMAGE
    mock_image.content = b'fake_image_data'

    return {
        'chapters': [mock_chapter1, mock_chapter2],
        'cover': mock_cover,
        'image': mock_image
    }


class TestEpubReader:
    """Test suite for EpubReader class."""

    @patch('audify.readers.ebook.epub.read_epub')
    def test_init_success(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test successful initialization of EpubReader."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)

        assert reader.path == temp_epub_path.resolve()
        assert reader.title == "Test Book Title"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_init_with_string_path(self, mock_read_epub, mock_epub_book):
        """Test initialization with string path."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader("test.epub")

        assert isinstance(reader.path, Path)
        assert reader.path == Path("test.epub").resolve()

    @patch('audify.readers.ebook.epub.read_epub')
    def test_read(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test reading EPUB content."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        assert reader.book == mock_epub_book
        mock_read_epub.assert_called_with(temp_epub_path)

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_chapters(
        self, mock_read_epub, temp_epub_path, mock_epub_book, mock_epub_items):
        """Test getting chapters from EPUB."""
        mock_epub_book.get_items.return_value = mock_epub_items['chapters'] + [
            mock_epub_items['cover']]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()
        chapters = reader.get_chapters()

        assert len(chapters) == 2
        assert chapters[0] == '<html><body><h1>Chapter 1</h1><p>' \
            'This is chapter 1 content.</p></body></html>'
        assert chapters[1] == '<html><body><h2>Chapter 2</h2><p>' \
            'This is chapter 2 content.</p></body></html>'

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_chapters_no_documents(
        self, mock_read_epub, temp_epub_path, mock_epub_book, mock_epub_items):
        """Test getting chapters when no document items exist."""
        mock_epub_book.get_items.return_value = [
            mock_epub_items['cover'], mock_epub_items['image']
        ]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()
        chapters = reader.get_chapters()

        assert chapters == []

    @patch('audify.readers.ebook.epub.read_epub')
    def test_extract_text(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test extracting text from HTML chapter content."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = '<html><body><h1>Chapter Title</h1>' \
            '<p>This is paragraph text.</p><div>More content</div></body></html>'

        text = reader.extract_text(html_content)

        expected_text = "Chapter TitleThis is paragraph text.More content"
        assert text == expected_text

    @patch('audify.readers.ebook.epub.read_epub')
    def test_extract_text_empty_html(
        self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test extracting text from empty HTML."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        text = reader.extract_text('<html><body></body></html>')

        assert text == ""

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_chapter_title_h1(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting chapter title from h1 tag."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = '<html><body><h1>Chapter One</h1>' \
            '<p>Content here</p></body></html>'

        title = reader.get_chapter_title(html_content)

        assert title == "Chapter One"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_chapter_title_h2(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting chapter title from h2 tag."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = '<html><body><h2>Chapter Two</h2>' \
            '<p>Content here</p></body></html>'

        title = reader.get_chapter_title(html_content)

        assert title == "Chapter Two"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_chapter_title_from_title_tag(
        self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting chapter title from title tag."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = '<html><head><title>Page Title</title>' \
            '</head><body><p>Content here</p></body></html>'

        title = reader.get_chapter_title(html_content)

        assert title == "Page Title"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_chapter_title_from_header(
        self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting chapter title from header tag."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = '<html><body><header>Header Title' \
            '</header><p>Content here</p></body></html>'

        title = reader.get_chapter_title(html_content)

        assert title == "Header Title"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_chapter_title_unknown(
        self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting chapter title when no title elements found."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = '<html><body><p>Just content, no title</p></body></html>'

        title = reader.get_chapter_title(html_content)

        assert title == "Unknown"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_title_from_book_title(self, mock_read_epub, temp_epub_path):
        """Test getting title from book.title attribute."""
        mock_book = Mock()
        mock_book.title = "Direct Book Title"
        mock_book.get_metadata.return_value = [("Metadata Title", {})]
        mock_read_epub.return_value = mock_book

        reader = EpubReader(temp_epub_path)

        assert reader.title == "Direct Book Title"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_title_from_metadata(self, mock_read_epub, temp_epub_path):
        """Test getting title from DC metadata when book.title is None."""
        mock_book = Mock()
        mock_book.title = None
        mock_book.get_metadata.return_value = [("Metadata Title", {})]
        mock_read_epub.return_value = mock_book

        reader = EpubReader(temp_epub_path)

        assert reader.title == "Metadata Title"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_title_missing(self, mock_read_epub, temp_epub_path):
        """Test getting title when both book.title and metadata are missing."""
        mock_book = Mock()
        mock_book.title = None
        mock_book.get_metadata.return_value = []
        mock_read_epub.return_value = mock_book

        reader = EpubReader(temp_epub_path)

        assert reader.title == "missing title"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_title_empty_metadata(self, mock_read_epub, temp_epub_path):
        """Test getting title when metadata exists but first element is empty."""
        mock_book = Mock()
        mock_book.title = None
        mock_book.get_metadata.return_value = [(None, {})]
        mock_read_epub.return_value = mock_book

        reader = EpubReader(temp_epub_path)

        assert reader.title == "missing title"

    @patch('builtins.open', new_callable=mock_open)
    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_cover_image_with_cover_item(
        self, mock_read_epub, mock_file_open,
        temp_epub_path, mock_epub_book, mock_epub_items):
        """Test getting cover image when ITEM_COVER exists."""
        mock_epub_book.get_items.return_value = [
            mock_epub_items['cover'], mock_epub_items['image']]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with tempfile.TemporaryDirectory() as temp_dir:
            cover_path = reader.get_cover_image(temp_dir)

            assert cover_path == Path(f"{temp_dir}/cover.jpg")
            mock_file_open.assert_called_with(f"{temp_dir}/cover.jpg", "wb")
            mock_file_open().write.assert_called_with(b'fake_cover_image_data')

    @patch('builtins.open', new_callable=mock_open)
    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_cover_image_fallback_to_first_image(
        self, mock_read_epub, mock_file_open,
        temp_epub_path, mock_epub_book, mock_epub_items):
        """Test getting cover image when no ITEM_COVER, fallback to first ITEM_IMAGE."""
        mock_epub_book.get_items.return_value = [mock_epub_items['image']]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with tempfile.TemporaryDirectory() as temp_dir:
            cover_path = reader.get_cover_image(temp_dir)

            assert cover_path == Path(f"{temp_dir}/cover.jpg")
            mock_file_open.assert_called_with(f"{temp_dir}/cover.jpg", "wb")
            mock_file_open().write.assert_called_with(b'fake_image_data')

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_cover_image_no_images(
        self, mock_read_epub, temp_epub_path, mock_epub_book, mock_epub_items):
        """Test getting cover image when no images exist."""
        mock_epub_book.get_items.return_value = mock_epub_items['chapters']
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with tempfile.TemporaryDirectory() as temp_dir:
            cover_path = reader.get_cover_image(temp_dir)

            assert cover_path is None

    @patch('builtins.open', new_callable=mock_open)
    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_cover_image_with_path_object(
        self, mock_read_epub, mock_file_open,
        temp_epub_path, mock_epub_book, mock_epub_items):
        """Test getting cover image with Path object as output_path."""
        mock_epub_book.get_items.return_value = [mock_epub_items['cover']]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            cover_path = reader.get_cover_image(output_path)

            assert cover_path == Path(f"{temp_dir}/cover.jpg")
            mock_file_open.assert_called_with(f"{temp_dir}/cover.jpg", "wb")

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_language(
        self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting language from metadata."""
        mock_epub_book.get_metadata.return_value = [("en", {})]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()
        language = reader.get_language()

        assert language == "en"
        mock_epub_book.get_metadata.assert_called_with("DC", "language")

    @patch('audify.readers.ebook.epub.read_epub')
    def test_get_language_multiple_languages(
        self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting language when multiple languages in metadata."""
        mock_epub_book.get_metadata.return_value = [("en", {}), ("fr", {})]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()
        language = reader.get_language()

        assert language == "en"  # Should return first language

    @patch('audify.readers.ebook.epub.read_epub')
    def test_epub_read_exception(self, mock_read_epub, temp_epub_path):
        """Test handling of ebooklib exceptions during EPUB reading."""
        mock_read_epub.side_effect = Exception("Corrupted EPUB file")

        with pytest.raises(Exception, match="Corrupted EPUB file"):
            EpubReader(temp_epub_path)

    @patch('audify.readers.ebook.epub.read_epub')
    def test_bs4_parsing_malformed_html(
        self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test BeautifulSoup handling of malformed HTML."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        malformed_html = '<html><body><h1>Unclosed tag<p>Missing closing tags'

        # BeautifulSoup should handle malformed HTML gracefully
        text = reader.extract_text(malformed_html)
        title = reader.get_chapter_title(malformed_html)

        assert "Unclosed tag" in text
        assert "Missing closing tags" in text
        assert title == "Unclosed tagMissing closing tags"

    @patch('audify.readers.ebook.epub.read_epub')
    def test_path_resolution(self, mock_read_epub, mock_epub_book):
        """Test that paths are properly resolved to absolute paths."""
        mock_read_epub.return_value = mock_epub_book

        # Test with relative path
        relative_path = "relative/path/file.epub"
        reader = EpubReader(relative_path)

        # Verify path was resolved to absolute
        assert reader.path.is_absolute()
        assert reader.path == Path(relative_path).resolve()

    @patch('audify.readers.ebook.epub.read_epub')
    def test_file_write_exception_handling(
        self, mock_read_epub, temp_epub_path, mock_epub_book, mock_epub_items):
        """Test handling of file write exceptions when saving cover image."""
        mock_epub_book.get_items.return_value = [mock_epub_items['cover']]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with tempfile.TemporaryDirectory() as temp_dir:
                with pytest.raises(IOError, match="Permission denied"):
                    reader.get_cover_image(temp_dir)
