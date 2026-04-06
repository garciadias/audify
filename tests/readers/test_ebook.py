import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import bs4
import pytest
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE

from audify.readers.ebook import EpubReader


@pytest.fixture
def temp_epub_path():
    """Create a temporary EPUB file path."""
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as temp_file:
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
    mock_chapter1.get_body_content.return_value = (
        b"<html><body><h1>Chapter 1</h1><p>This is chapter 1 content.</p></body></html>"
    )

    mock_chapter2 = Mock()
    mock_chapter2.get_type.return_value = ITEM_DOCUMENT
    mock_chapter2.get_body_content.return_value = (
        b"<html><body><h2>Chapter 2</h2><p>This is chapter 2 content.</p></body></html>"
    )

    # Mock cover item
    mock_cover = Mock()
    mock_cover.get_type.return_value = ITEM_COVER
    mock_cover.content = b"fake_cover_image_data"

    # Mock image item
    mock_image = Mock()
    mock_image.get_type.return_value = ITEM_IMAGE
    mock_image.content = b"fake_image_data"

    return {
        "chapters": [mock_chapter1, mock_chapter2],
        "cover": mock_cover,
        "image": mock_image,
    }


class TestEpubReader:
    """Test suite for EpubReader class."""

    @patch("audify.readers.ebook.epub.read_epub")
    def test_init_success(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test successful initialization of EpubReader."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)

        assert reader.path == temp_epub_path.resolve()
        assert reader.title == "Test Book Title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_init_with_string_path(self, mock_read_epub, mock_epub_book):
        """Test initialization with string path."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader("test.epub")

        assert isinstance(reader.path, Path)
        assert reader.path == Path("test.epub").resolve()

    @patch("audify.readers.ebook.epub.read_epub")
    def test_read(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test reading EPUB content."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        assert reader.book == mock_epub_book
        mock_read_epub.assert_called_with(temp_epub_path)

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapters(
        self, mock_read_epub, temp_epub_path, mock_epub_book, mock_epub_items
    ):
        """Test getting chapters from EPUB."""
        mock_epub_book.get_items.return_value = mock_epub_items["chapters"] + [
            mock_epub_items["cover"]
        ]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()
        chapters = reader.get_chapters()

        assert len(chapters) == 2
        assert (
            chapters[0] == "<html><body><h1>Chapter 1</h1><p>"
            "This is chapter 1 content.</p></body></html>"
        )
        assert (
            chapters[1] == "<html><body><h2>Chapter 2</h2><p>"
            "This is chapter 2 content.</p></body></html>"
        )

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapters_no_documents(
        self, mock_read_epub, temp_epub_path, mock_epub_book, mock_epub_items
    ):
        """Test getting chapters when no document items exist."""
        mock_epub_book.get_items.return_value = [
            mock_epub_items["cover"],
            mock_epub_items["image"],
        ]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()
        chapters = reader.get_chapters()

        assert chapters == []

    @patch("audify.readers.ebook.epub.read_epub")
    def test_extract_text(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test extracting text from HTML chapter content."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><h1>Chapter Title</h1>"
            "<p>This is paragraph text.</p><div>More content</div></body></html>"
        )

        text = reader.extract_text(html_content)

        expected_text = "Chapter TitleThis is paragraph text.More content"
        assert text == expected_text

    @patch("audify.readers.ebook.epub.read_epub")
    def test_extract_text_empty_html(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting text from empty HTML."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        text = reader.extract_text("<html><body></body></html>")

        assert text == ""

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_h1(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting chapter title from h1 tag."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><h1>Chapter One</h1><p>Content here</p></body></html>"
        )

        title = reader.get_chapter_title(html_content)

        assert title == "Chapter One"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_h2(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting chapter title from h2 tag."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><h2>Chapter Two</h2><p>Content here</p></body></html>"
        )

        title = reader.get_chapter_title(html_content)

        assert title == "Chapter Two"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_from_title_tag(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test getting chapter title from title tag."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><head><title>Page Title</title>"
            "</head><body><p>Content here</p></body></html>"
        )

        title = reader.get_chapter_title(html_content)

        assert title == "Page Title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_from_header(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test getting chapter title from header tag."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><header>Header Title</header><p>Content here</p></body></html>"
        )

        title = reader.get_chapter_title(html_content)

        assert title == "Header Title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_unknown(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test getting chapter title when no title elements found."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        # All paragraphs are long sentences ending with periods — no title found
        html_content = (
            "<html><body>"
            "<p>This is a long paragraph that ends with a period and "
            "clearly is not a title because it is too long and descriptive.</p>"
            "<p>Another paragraph with regular content that goes on and on.</p>"
            "</body></html>"
        )

        title = reader.get_chapter_title(html_content)

        assert title == "Unknown"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_from_class_attribute(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting title from element with title-related CSS class."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            '<html><body><div class="chapter-title">The Dark Forest</div>'
            "<p>Content here</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "The Dark Forest"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_from_id_attribute(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting title from element with title-related id."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            '<html><body><div id="chaptertitle">A New Beginning</div>'
            "<p>Content here</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "A New Beginning"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_from_calibre_heading(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting title from Calibre-generated heading class."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            '<html><body><p class="calibre_heading">Midnight Sun</p>'
            "<p>The story begins...</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Midnight Sun"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_regex_chapter_number(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting 'Chapter N' pattern from a paragraph."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><p>Chapter 3: The Journey Begins</p>"
            "<p>They set off at dawn.</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Chapter 3: The Journey Begins"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_regex_part(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting 'Part N' pattern."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><p>Part II - The Return</p>"
            "<p>Content follows here.</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Part II - The Return"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_regex_prologue(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting 'Prologue' pattern."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><p>Prologue</p>"
            "<p>A long time ago in a land far away...</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Prologue"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_regex_epilogue(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting 'Epilogue' pattern."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><p>Epilogue: Aftermath</p><p>Years later...</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Epilogue: Aftermath"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_short_paragraph(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting title from a short paragraph that looks like a title."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><p>The Great Escape</p>"
            "<p>It was a dark and stormy night when the prisoners made their move.</p>"
            "</body></html>"
        )
        assert reader.get_chapter_title(html_content) == "The Great Escape"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_emphasis_bold_chapter(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test extracting title from bold text with chapter pattern."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><p><strong>Chapter 7 — The Reckoning</strong></p>"
            "<p>The day had finally come.</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Chapter 7 — The Reckoning"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_heading_takes_priority(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that heading tags take priority over other strategies."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body>"
            "<h2>Real Title</h2>"
            '<p class="chapter-title">CSS Title</p>'
            "<p>Chapter 1: Paragraph Title</p>"
            "</body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Real Title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_llm_fallback(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test LLM fallback when no other strategy works."""
        mock_read_epub.return_value = mock_epub_book

        mock_llm = Mock()
        mock_llm.generate.return_value = "The Forgotten Kingdom"

        reader = EpubReader(temp_epub_path, llm_config=mock_llm)
        # Long sentences with periods — no heading, no class, no pattern, no short title
        html_content = (
            "<html><body>"
            "<p>This is a fairly long first paragraph that clearly "
            "is not a title because it ends with a period.</p>"
            "<p>And this second paragraph is also regular content with a period.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "The Forgotten Kingdom"
        mock_llm.generate.assert_called_once()

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_llm_not_called_when_heading_found(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that LLM is not called when heading tag provides the title."""
        mock_read_epub.return_value = mock_epub_book

        mock_llm = Mock()
        reader = EpubReader(temp_epub_path, llm_config=mock_llm)
        html_content = "<html><body><h1>Found Title</h1><p>Content</p></body></html>"
        title = reader.get_chapter_title(html_content)
        assert title == "Found Title"
        mock_llm.generate.assert_not_called()

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_llm_error_returns_unknown(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that LLM errors fall back to Unknown."""
        mock_read_epub.return_value = mock_epub_book

        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("Connection refused")

        reader = EpubReader(temp_epub_path, llm_config=mock_llm)
        html_content = (
            "<html><body>"
            "<p>This is a fairly long first paragraph that clearly "
            "is not a title because it ends with a period.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "Unknown"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_llm_returns_unknown_string(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that LLM returning 'Unknown' is treated as no result."""
        mock_read_epub.return_value = mock_epub_book

        mock_llm = Mock()
        mock_llm.generate.return_value = "Unknown"

        reader = EpubReader(temp_epub_path, llm_config=mock_llm)
        html_content = (
            "<html><body>"
            "<p>This is a fairly long first paragraph that clearly "
            "is not a title because it ends with a period.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "Unknown"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_llm_reasoning_model(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that <think> tags from reasoning models are stripped."""
        mock_read_epub.return_value = mock_epub_book

        mock_llm = Mock()
        mock_llm.generate.return_value = (
            "<think>Let me analyze...</think>The Hidden Valley"
        )

        reader = EpubReader(temp_epub_path, llm_config=mock_llm)
        html_content = (
            "<html><body>"
            "<p>This is a fairly long first paragraph that clearly "
            "is not a title because it ends with a period.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "The Hidden Valley"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_no_llm_config_returns_unknown(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that without LLM config, unresolvable chapters return Unknown."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        assert reader.llm_config is None
        html_content = (
            "<html><body>"
            "<p>This is a fairly long first paragraph that clearly "
            "is not a title because it ends with a period.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "Unknown"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_roman_numeral_with_subtitle(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test Roman numeral pattern with subtitle."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><p>IV - The Last Stand</p><p>Content here.</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "IV - The Last Stand"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_title_from_book_title(self, mock_read_epub, temp_epub_path):
        """Test getting title from book.title attribute."""
        mock_book = Mock()
        mock_book.title = "Direct Book Title"
        mock_book.get_metadata.return_value = [("Metadata Title", {})]
        mock_read_epub.return_value = mock_book

        reader = EpubReader(temp_epub_path)

        assert reader.title == "Direct Book Title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_title_from_metadata(self, mock_read_epub, temp_epub_path):
        """Test getting title from DC metadata when book.title is None."""
        mock_book = Mock()
        mock_book.title = None
        mock_book.get_metadata.return_value = [("Metadata Title", {})]
        mock_read_epub.return_value = mock_book

        reader = EpubReader(temp_epub_path)

        assert reader.title == "Metadata Title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_title_missing(self, mock_read_epub, temp_epub_path):
        """Test getting title when both book.title and metadata are missing."""
        mock_book = Mock()
        mock_book.title = None
        mock_book.get_metadata.return_value = []
        mock_read_epub.return_value = mock_book

        reader = EpubReader(temp_epub_path)

        assert reader.title == "missing title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_title_empty_metadata(self, mock_read_epub, temp_epub_path):
        """Test getting title when metadata exists but first element is empty."""
        mock_book = Mock()
        mock_book.title = None
        mock_book.get_metadata.return_value = [(None, {})]
        mock_read_epub.return_value = mock_book

        reader = EpubReader(temp_epub_path)

        assert reader.title == "missing title"

    @patch("builtins.open", new_callable=mock_open)
    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_cover_image_with_cover_item(
        self,
        mock_read_epub,
        mock_file_open,
        temp_epub_path,
        mock_epub_book,
        mock_epub_items,
    ):
        """Test getting cover image when ITEM_COVER exists."""
        mock_epub_book.get_items.return_value = [
            mock_epub_items["cover"],
            mock_epub_items["image"],
        ]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with tempfile.TemporaryDirectory() as temp_dir:
            cover_path = reader.get_cover_image(temp_dir)

            assert cover_path == Path(f"{temp_dir}/cover.jpg")
            mock_file_open.assert_called_with(f"{temp_dir}/cover.jpg", "wb")
            mock_file_open().write.assert_called_with(b"fake_cover_image_data")

    @patch("builtins.open", new_callable=mock_open)
    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_cover_image_fallback_to_first_image(
        self,
        mock_read_epub,
        mock_file_open,
        temp_epub_path,
        mock_epub_book,
        mock_epub_items,
    ):
        """Test getting cover image when no ITEM_COVER, fallback to first ITEM_IMAGE."""
        mock_epub_book.get_items.return_value = [mock_epub_items["image"]]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with tempfile.TemporaryDirectory() as temp_dir:
            cover_path = reader.get_cover_image(temp_dir)

            assert cover_path == Path(f"{temp_dir}/cover.jpg")
            mock_file_open.assert_called_with(f"{temp_dir}/cover.jpg", "wb")
            mock_file_open().write.assert_called_with(b"fake_image_data")

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_cover_image_no_images(
        self, mock_read_epub, temp_epub_path, mock_epub_book, mock_epub_items
    ):
        """Test getting cover image when no images exist."""
        mock_epub_book.get_items.return_value = mock_epub_items["chapters"]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with tempfile.TemporaryDirectory() as temp_dir:
            cover_path = reader.get_cover_image(temp_dir)

            assert cover_path is None

    @patch("builtins.open", new_callable=mock_open)
    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_cover_image_with_path_object(
        self,
        mock_read_epub,
        mock_file_open,
        temp_epub_path,
        mock_epub_book,
        mock_epub_items,
    ):
        """Test getting cover image with Path object as output_path."""
        mock_epub_book.get_items.return_value = [mock_epub_items["cover"]]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            cover_path = reader.get_cover_image(output_path)

            assert cover_path == Path(f"{temp_dir}/cover.jpg")
            mock_file_open.assert_called_with(f"{temp_dir}/cover.jpg", "wb")

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_language(self, mock_read_epub, temp_epub_path, mock_epub_book):
        """Test getting language from metadata."""
        mock_epub_book.get_metadata.return_value = [("en", {})]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()
        language = reader.get_language()

        assert language == "en"
        mock_epub_book.get_metadata.assert_called_with("DC", "language")

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_language_multiple_languages(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test getting language when multiple languages in metadata."""
        mock_epub_book.get_metadata.return_value = [("en", {}), ("fr", {})]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()
        language = reader.get_language()

        assert language == "en"  # Should return first language

    @patch("audify.readers.ebook.epub.read_epub")
    def test_epub_read_exception(self, mock_read_epub, temp_epub_path):
        """Test handling of ebooklib exceptions during EPUB reading."""
        mock_read_epub.side_effect = Exception("Corrupted EPUB file")

        with pytest.raises(Exception, match="Corrupted EPUB file"):
            EpubReader(temp_epub_path)

    @patch("audify.readers.ebook.epub.read_epub")
    def test_bs4_parsing_malformed_html(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test BeautifulSoup handling of malformed HTML."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        malformed_html = "<html><body><h1>Unclosed tag<p>Missing closing tags"

        # BeautifulSoup should handle malformed HTML gracefully
        text = reader.extract_text(malformed_html)
        title = reader.get_chapter_title(malformed_html)

        assert "Unclosed tag" in text
        assert "Missing closing tags" in text
        assert "Unclosed tag" in title

    @patch("audify.readers.ebook.epub.read_epub")
    def test_path_resolution(self, mock_read_epub, mock_epub_book):
        """Test that paths are properly resolved to absolute paths."""
        mock_read_epub.return_value = mock_epub_book

        # Test with relative path
        relative_path = "relative/path/file.epub"
        reader = EpubReader(relative_path)

        # Verify path was resolved to absolute
        assert reader.path.is_absolute()
        assert reader.path == Path(relative_path).resolve()

    @patch("audify.readers.ebook.epub.read_epub")
    def test_file_write_exception_handling(
        self, mock_read_epub, temp_epub_path, mock_epub_book, mock_epub_items
    ):
        """Test handling of file write exceptions when saving cover image."""
        mock_epub_book.get_items.return_value = [mock_epub_items["cover"]]
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        reader.read()

        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with tempfile.TemporaryDirectory() as temp_dir:
                with pytest.raises(IOError, match="Permission denied"):
                    reader.get_cover_image(temp_dir)

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_no_body_tag(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test title extraction from HTML without a <body> tag."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = "<p>Chapter 1: No Body Tag</p><p>Content here.</p>"

        title = reader.get_chapter_title(html_content)
        assert title == "Chapter 1: No Body Tag"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_empty_paragraphs_skipped(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that empty paragraphs are skipped during extraction."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body>"
            "<p></p><p>   </p>"
            "<p>Chapter 5: After Empty Paragraphs</p>"
            "<p>Some regular content follows here.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "Chapter 5: After Empty Paragraphs"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_emphasis_without_heading_or_class(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test bold emphasis tag extraction when no heading or class matches."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        # No headings, no title-related class/id, but bold text with chapter pattern
        html_content = (
            "<html><body>"
            "<p><b>Chapter 12 — The Awakening</b></p>"
            "<p>The sun rose over the mountains and everything changed.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "Chapter 12 — The Awakening"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_llm_with_empty_body(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test LLM fallback when body has no extractable text."""
        mock_read_epub.return_value = mock_epub_book

        mock_llm = Mock()
        reader = EpubReader(temp_epub_path, llm_config=mock_llm)
        # All paragraphs are empty — LLM should not be called
        html_content = "<html><body><p>   </p><p></p></body></html>"
        title = reader.get_chapter_title(html_content)
        assert title == "Unknown"
        mock_llm.generate.assert_not_called()

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_llm_strips_quotes(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that LLM response with quotes is properly stripped."""
        mock_read_epub.return_value = mock_epub_book

        mock_llm = Mock()
        mock_llm.generate.return_value = '"The Winding Road"'

        reader = EpubReader(temp_epub_path, llm_config=mock_llm)
        html_content = (
            "<html><body>"
            "<p>This is a fairly long first paragraph that clearly "
            "is not a title because it ends with a period.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "The Winding Road"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_llm_empty_response(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that empty LLM response falls back to Unknown."""
        mock_read_epub.return_value = mock_epub_book

        mock_llm = Mock()
        mock_llm.generate.return_value = ""

        reader = EpubReader(temp_epub_path, llm_config=mock_llm)
        html_content = (
            "<html><body>"
            "<p>This is a fairly long first paragraph that clearly "
            "is not a title because it ends with a period.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "Unknown"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_class_with_empty_text(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that title-class elements with empty text are skipped."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body>"
            '<div class="chapter-title">   </div>'
            "<p>Actual Title Here</p>"
            "<p>Then some long content that follows after the title paragraph.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "Actual Title Here"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapter_title_heading_empty_text_falls_through(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test that heading tags with empty text fall through to next strategy."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body><h1>   </h1><p>Prologue: The Beginning</p></body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "Prologue: The Beginning"

    # --- Regression tests for review findings ---

    @patch("audify.readers.ebook.epub.read_epub")
    def test_heading_body_h1_preferred_over_head_title(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Regression: <h1> in body should win over <title> in <head>."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><head><title>Page Title</title></head>"
            "<body><h1>Body Heading</h1><p>Content.</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Body Heading"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_heading_title_fallback_when_no_body_heading(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Regression: <title> should still work when no heading in body."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><head><title>Fallback Title</title></head>"
            "<body><p>Content only, no headings here.</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Fallback Title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_title_attr_no_false_positive_on_section_content(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Regression: class='section-content' should NOT match title pattern."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        # The div has a non-title class; the content is long enough
        # and ends with a period to avoid matching short-paragraph heuristic
        html_content = (
            "<html><body>"
            '<div class="section-content">'
            "This is a long paragraph inside a section-content div "
            "that should not be matched as a title at all.</div>"
            "<p>Another long paragraph to fill the body "
            "with regular text content.</p>"
            "</body></html>"
        )
        # Should NOT match "section-content" as a title class
        title = reader.get_chapter_title(html_content)
        assert title == "Unknown"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_title_attr_true_positive_still_matches(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Regression: class='chapter-title' should still match."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body>"
            '<div class="chapter-title">The Real Title</div>'
            "<p>Content.</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "The Real Title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_nested_wrapper_returns_leaf_not_container(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Regression: nested <div> wrapper should not return combined text."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        # Use a non-title class so strategy 2 doesn't match the wrapper
        html_content = (
            "<html><body>"
            '<div class="content"><p>Chapter 3</p>'
            "<p>Body text that is long enough to not look like a "
            "title and ends with a period.</p></div>"
            "</body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Chapter 3"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_title_attr_matches_token_at_start(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Regression: class starting with a title token should match."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body>"
            '<p class="title-text">Opening Title</p>'
            "<p>Content.</p></body></html>"
        )
        assert reader.get_chapter_title(html_content) == "Opening Title"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_title_attr_no_false_positive_on_subtitle(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Regression: class='subtitle' should NOT match (embedded 'title')."""
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        html_content = (
            "<html><body>"
            '<div class="subtitle">'
            "This is a long subtitle paragraph that should not "
            "be matched as a title by the CSS class pattern.</div>"
            "<p>Another long paragraph with regular content "
            "that fills the body.</p>"
            "</body></html>"
        )
        title = reader.get_chapter_title(html_content)
        assert title == "Unknown"

    @patch("audify.readers.ebook.epub.read_epub")
    def test_extract_from_title_attributes_with_non_tag(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test _extract_from_title_attributes handles non-Tag elements."""
        mock_read_epub.return_value = mock_epub_book
        reader = EpubReader(temp_epub_path)
        soup = bs4.BeautifulSoup("<html><body>Text</body></html>", "html.parser")
        # Mock find_all to return a non-Tag
        non_tag = bs4.element.NavigableString("test")
        with patch.object(soup, "find_all") as mock_find_all:
            mock_find_all.return_value = [non_tag]
            result = reader._extract_from_title_attributes(soup)
            assert result == ""

    @patch("audify.readers.ebook.epub.read_epub")
    def test_extract_from_emphasis_tags_with_non_tag(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test _extract_from_emphasis_tags handles non-Tag children."""
        mock_read_epub.return_value = mock_epub_book
        reader = EpubReader(temp_epub_path)
        # Soup where body has a NavigableString child (text node)
        soup = bs4.BeautifulSoup("<html><body>Text</body></html>", "html.parser")
        result = reader._extract_from_emphasis_tags(soup)
        assert result == ""

    @patch("audify.readers.ebook.epub.read_epub")
    def test_is_leaf_paragraph_with_non_tag(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test _is_leaf_paragraph returns True for non-Tag."""
        mock_read_epub.return_value = mock_epub_book
        reader = EpubReader(temp_epub_path)
        non_tag = bs4.element.NavigableString("test")
        result = reader._is_leaf_paragraph(non_tag)
        assert result is True

    @patch("audify.readers.ebook.epub.read_epub")
    def test_extract_short_paragraph_title_with_nested_paragraph(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Test _extract_short_paragraph_title skips nested paragraphs."""
        mock_read_epub.return_value = mock_epub_book
        reader = EpubReader(temp_epub_path)
        # Create a div containing a p element with long text ending with period
        soup = bs4.BeautifulSoup(
            "<html><body><div><p>This is a long paragraph that definitely exceeds "
            "eighty characters and ends with a period.</p></div></body></html>",
            "html.parser",
        )
        result = reader._extract_short_paragraph_title(soup)
        # Should skip the div (nested) and the p is too long
        assert result == ""
