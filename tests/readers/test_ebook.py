import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import bs4
import pytest
from ebooklib import ITEM_COVER, ITEM_DOCUMENT, ITEM_IMAGE
from ebooklib.epub import Link

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
    mock_book.spine = []  # Empty spine to fall back to all items
    mock_book.get_item_with_id = Mock(return_value=None)
    mock_book.toc = []  # Explicit empty TOC (Mock.toc returns a non-list Mock)
    return mock_book


@pytest.fixture
def mock_epub_items():
    """Create mock EPUB items (chapters, images, covers)."""
    # Mock document item (chapter)
    mock_chapter1 = Mock()
    mock_chapter1.id = "section1"
    mock_chapter1.get_type.return_value = ITEM_DOCUMENT
    mock_chapter1.get_name.return_value = "section1.xhtml"
    mock_chapter1.get_body_content.return_value = (
        b"<html><body><h1>Section 1</h1><p>"
        b"This is section 1 content. This is a longer paragraph to exceed 100 "
        b"characters. We need to ensure that the text length after extraction is "
        b"sufficient to pass the filter. Adding more text here to be safe."
        b"</p></body></html>"
    )

    mock_chapter2 = Mock()
    mock_chapter2.id = "section2"
    mock_chapter2.get_type.return_value = ITEM_DOCUMENT
    mock_chapter2.get_name.return_value = "section2.xhtml"
    mock_chapter2.get_body_content.return_value = (
        b"<html><body><h2>Section 2</h2><p>"
        b"This is section 2 content. This is a longer paragraph to exceed 100 "
        b"characters. We need to ensure that the text length after extraction is "
        b"sufficient to pass the filter. Adding more text here to be safe."
        b"</p></body></html>"
    )

    # Mock cover item
    mock_cover = Mock()
    mock_cover.id = "cover"
    mock_cover.get_type.return_value = ITEM_COVER
    mock_cover.get_name.return_value = "cover.jpg"
    mock_cover.content = b"fake_cover_image_data"

    # Mock image item
    mock_image = Mock()
    mock_image.id = "image1"
    mock_image.get_type.return_value = ITEM_IMAGE
    mock_image.get_name.return_value = "image1.png"
    mock_image.content = b"fake_image_data"

    return {
        "chapters": [mock_chapter1, mock_chapter2],
        "cover": mock_cover,
        "image": mock_image,
    }


class TestEpubReader:
    """Test suite for EpubReader class."""

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
            chapters[0] == "<html><body><h1>Section 1</h1><p>"
            "This is section 1 content. This is a longer paragraph to exceed 100 "
            "characters. "
            "We need to ensure that the text length after extraction is sufficient "
            "to pass the filter. "
            "Adding more text here to be safe.</p></body></html>"
        )
        assert (
            chapters[1] == "<html><body><h2>Section 2</h2><p>"
            "This is section 2 content. This is a longer paragraph to exceed 100 "
            "characters. "
            "We need to ensure that the text length after extraction is sufficient "
            "to pass the filter. "
            "Adding more text here to be safe.</p></body></html>"
        )


    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapters_skips_multilingual_front_matter_by_filename(
        self,
        mock_read_epub,
        temp_epub_path,
        mock_epub_book,
        mock_epub_items,
    ):
        """Skip common non-chapter files like cubierta/titulo/notas."""
        mock_cubierta = Mock()
        mock_cubierta.get_type.return_value = ITEM_DOCUMENT
        mock_cubierta.get_name.return_value = "Text/cubierta.xhtml"
        mock_cubierta.get_body_content.return_value = (
            b"<html><body><p>" + b"x" * 300 + b"</p></body></html>"
        )

        mock_titulo = Mock()
        mock_titulo.get_type.return_value = ITEM_DOCUMENT
        mock_titulo.get_name.return_value = "Text/titulo.xhtml"
        mock_titulo.get_body_content.return_value = (
            b"<html><body><p>" + b"x" * 300 + b"</p></body></html>"
        )

        mock_notas = Mock()
        mock_notas.get_type.return_value = ITEM_DOCUMENT
        mock_notas.get_name.return_value = "Text/notas.xhtml"
        mock_notas.get_body_content.return_value = (
            b"<html><body><p>" + b"x" * 300 + b"</p></body></html>"
        )

        chapter_item = mock_epub_items["chapters"][0]

        mock_epub_book.spine = [
            ("c1", "yes"),
            ("c2", "yes"),
            ("c3", "yes"),
            ("c4", "yes"),
        ]

        id_map = {
            "c1": mock_cubierta,
            "c2": mock_titulo,
            "c3": chapter_item,
            "c4": mock_notas,
        }
        mock_epub_book.get_item_with_id.side_effect = lambda x: id_map.get(x)
        mock_epub_book.get_items.return_value = list(id_map.values())
        mock_read_epub.return_value = mock_epub_book

        reader = EpubReader(temp_epub_path)
        chapters = reader.get_chapters()

        assert len(chapters) == 1
        assert "Section 1" in chapters[0]

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

    # --- Regression tests for review findings ---

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


class TestEpubReaderTocGrouping:
    """TOC-based chapter grouping unit tests."""

    # ------------------------------------------------------------------
    # _flatten_toc_hrefs
    # ------------------------------------------------------------------

    @patch("audify.readers.ebook.epub.read_epub")
    def test_flatten_toc_hrefs_nested(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Nested (Link, [Link]) tuples are flattened into a single list."""
        mock_read_epub.return_value = mock_epub_book
        reader = EpubReader(temp_epub_path)
        reader.book.toc = [
            (
                Link("Text/part1.xhtml", "Part 1"),
                [
                    Link("Text/ch1.xhtml", "Ch 1"),
                    Link("Text/ch2.xhtml", "Ch 2"),
                ],
            ),
            Link("Text/ch3.xhtml", "Ch 3"),
        ]
        assert reader._flatten_toc_hrefs() == [
            "Text/part1.xhtml",
            "Text/ch1.xhtml",
            "Text/ch2.xhtml",
            "Text/ch3.xhtml",
        ]


    @patch("audify.readers.ebook.epub.read_epub")
    def test_flatten_toc_hrefs_non_list(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """TOC attribute that is not a list returns empty list."""
        mock_read_epub.return_value = mock_epub_book
        reader = EpubReader(temp_epub_path)
        reader.book.toc = None
        assert reader._flatten_toc_hrefs() == []

    # ------------------------------------------------------------------
    # _build_toc_item_name_set
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # _merge_items
    # ------------------------------------------------------------------

    @patch("audify.readers.ebook.epub.read_epub")
    def test_merge_items_empty(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Empty items list returns None."""
        mock_read_epub.return_value = mock_epub_book
        reader = EpubReader(temp_epub_path)
        assert reader._merge_items([]) is None

    @patch("audify.readers.ebook.epub.read_epub")
    def test_merge_items_decode_failure(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Item with undecodable body is skipped in a multi-item merge."""
        mock_read_epub.return_value = mock_epub_book
        reader = EpubReader(temp_epub_path)
        bad_item = Mock()
        bad_item.get_body_content.side_effect = Exception("decode error")
        bad_item.get_name.return_value = "bad.xhtml"
        good_item = Mock()
        good_item.get_body_content.return_value = b"<html><body>Good</body></html>"
        good_item.get_name.return_value = "good.xhtml"
        result = reader._merge_items([bad_item, good_item])
        assert result is not None
        assert "Good" in result

    # ------------------------------------------------------------------
    # _looks_like_toc (classmethod)
    # ------------------------------------------------------------------

    @patch("audify.readers.ebook.epub.read_epub")
    def test_looks_like_toc_css_class(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """TOC identified by CSS class 'toc' with many links."""
        mock_read_epub.return_value = mock_epub_book
        html = (
            "<html><body>"
            "<div class='toc'>"
            "<ul>"
            "<li><a href='#c1'>Chapter 1</a></li>"
            "<li><a href='#c2'>Chapter 2</a></li>"
            "<li><a href='#c3'>Chapter 3</a></li>"
            "<li><a href='#c4'>Chapter 4</a></li>"
            "<li><a href='#c5'>Chapter 5</a></li>"
            "</ul>"
            "</div>"
            "</body></html>"
        )
        soup = bs4.BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True).lower()
        assert EpubReader._looks_like_toc(soup, text) is True


    @patch("audify.readers.ebook.epub.read_epub")
    def test_looks_like_toc_high_link_density(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """Content with no heading but very high link-to-text ratio is TOC."""
        mock_read_epub.return_value = mock_epub_book
        links = "".join(f"<li><a href='#c{i}'>Item {i}</a></li>" for i in range(25))
        html = (
            "<html><body>"
            f"<ul>{links}</ul>"
            "</body></html>"
        )
        soup = bs4.BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True).lower()
        assert EpubReader._looks_like_toc(soup, text) is True

    # ------------------------------------------------------------------
    # _looks_like_copyright (static)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # _is_valid_chapter
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # _get_chapters_grouped_by_toc
    # ------------------------------------------------------------------

    @patch("audify.readers.ebook.epub.read_epub")
    def test_get_chapters_grouped_by_toc_single_chapter_blob_check(
        self, mock_read_epub, temp_epub_path, mock_epub_book
    ):
        """1 chapter from many TOC entries triggers fallback (single-blob guard)."""
        mock_read_epub.return_value = mock_epub_book
        reader = EpubReader(temp_epub_path)

        reader.book.toc = [
            Link("Text/ch1.xhtml", "Ch 1"),
            Link("Text/ch2.xhtml", "Ch 2"),
            Link("Text/ch3.xhtml", "Ch 3"),
        ]

        ch1 = Mock()
        ch1.get_name.return_value = "Text/ch1.xhtml"
        ch1.get_type.return_value = ITEM_DOCUMENT
        ch1.get_body_content.return_value = (
            b"<html><body><p>" + b"Valid content " * 20 + b"</p></body></html>"
        )
        sub1 = Mock()
        sub1.get_name.return_value = "Text/sub1.xhtml"
        sub1.get_type.return_value = ITEM_DOCUMENT
        sub1.get_body_content.return_value = (
            b"<html><body><p>" + b"More valid content " * 20 + b"</p></body></html>"
        )

        reader.book.spine = [("ch1_id", "yes"), ("sub1_id", "yes")]
        reader.book.get_item_with_id.side_effect = lambda sid: {
            "ch1_id": ch1,
            "sub1_id": sub1,
        }.get(sid)

        chapters = reader._get_chapters_grouped_by_toc()
        assert chapters == []
