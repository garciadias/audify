import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import PyPDF2
import pytest

from audify.readers.pdf import PdfReader


@pytest.fixture
def temp_pdf_path():
    """Create a temporary PDF file path."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_pdf_content():
    """Mock PDF content for testing."""
    return "This is sample PDF text content.\n" \
           "With multiple lines.\nAnd some special characters: àáâã"


@pytest.fixture
def mock_cleaned_text():
    """Mock cleaned text output."""
    return "This is sample PDF text content. With multiple lines." \
           " And some special characters: àáâã"


class TestPdfReader:
    """Test suite for PdfReader class."""

    @patch('audify.readers.pdf.clean_text')
    @patch('audify.readers.pdf.PyPDF2.PdfReader')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_success(
            self,
            mock_file_open,
            mock_exists,
            mock_pdf_reader,
            mock_clean_text,
            temp_pdf_path
        ):
        """Test successful initialization of PdfReader."""
        # Setup mocks
        mock_exists.return_value = True
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF text"
        mock_pdf_reader_instance = Mock()
        mock_pdf_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_clean_text.return_value = "Cleaned sample PDF text"

        # Test initialization with Path object
        reader = PdfReader(temp_pdf_path)

        # Assertions
        assert reader.path == temp_pdf_path.resolve()
        assert reader.text == "Sample PDF text"
        assert reader.cleaned_text == "Cleaned sample PDF text"
        mock_exists.assert_called_once()
        mock_pdf_reader.assert_called_once()
        mock_clean_text.assert_called_once_with("Sample PDF text")

    @patch('pathlib.Path.exists')
    def test_init_file_not_found(self, mock_exists):
        """Test FileNotFoundError when PDF file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="PDF file not found at"):
            PdfReader("/nonexistent/path/file.pdf")

    @patch('audify.readers.pdf.clean_text')
    @patch('audify.readers.pdf.PyPDF2.PdfReader')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_init_with_string_path(
        self, mock_file_open, mock_exists, mock_pdf_reader, mock_clean_text):
        """Test initialization with string path."""
        mock_exists.return_value = True
        mock_page = Mock()
        mock_page.extract_text.return_value = "Text from string path"
        mock_pdf_reader_instance = Mock()
        mock_pdf_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_clean_text.return_value = "Cleaned text"

        reader = PdfReader("test.pdf")

        assert isinstance(reader.path, Path)
        assert reader.text == "Text from string path"

    @patch('audify.readers.pdf.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open)
    def test_read_single_page(self, mock_file_open, mock_pdf_reader, temp_pdf_path):
        """Test reading text from a single-page PDF."""
        # Setup mock for single page
        mock_page = Mock()
        mock_page.extract_text.return_value = "Single page content"
        mock_pdf_reader_instance = Mock()
        mock_pdf_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_pdf_reader_instance

        with patch('pathlib.Path.exists', return_value=True):
            with patch('audify.readers.pdf.clean_text', return_value="cleaned"):
                reader = PdfReader(temp_pdf_path)
                result = reader.read()

        assert result == "Single page content"
        mock_file_open.assert_called_with(temp_pdf_path, "rb")
        mock_pdf_reader.assert_called()

    @patch('audify.readers.pdf.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open)
    def test_read_multiple_pages(self, mock_file_open, mock_pdf_reader, temp_pdf_path):
        """Test reading text from a multi-page PDF."""
        # Setup mocks for multiple pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_page3 = Mock()
        mock_page3.extract_text.return_value = "Page 3 content"

        mock_pdf_reader_instance = Mock()
        mock_pdf_reader_instance.pages = [mock_page1, mock_page2, mock_page3]
        mock_pdf_reader.return_value = mock_pdf_reader_instance

        with patch('pathlib.Path.exists', return_value=True):
            with patch('audify.readers.pdf.clean_text', return_value="cleaned"):
                reader = PdfReader(temp_pdf_path)
                result = reader.read()

        expected_text = "Page 1 contentPage 2 contentPage 3 content"
        assert result == expected_text

    @patch('audify.readers.pdf.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open)
    def test_read_empty_pdf(self, mock_file_open, mock_pdf_reader, temp_pdf_path):
        """Test reading from an empty PDF (no pages)."""
        mock_pdf_reader_instance = Mock()
        mock_pdf_reader_instance.pages = []
        mock_pdf_reader.return_value = mock_pdf_reader_instance

        with patch('pathlib.Path.exists', return_value=True):
            with patch('audify.readers.pdf.clean_text', return_value=""):
                reader = PdfReader(temp_pdf_path)
                result = reader.read()

        assert result == ""

    @patch('audify.readers.pdf.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open)
    def test_read_pdf_with_special_characters(
        self, mock_file_open, mock_pdf_reader, temp_pdf_path):
        """Test reading PDF with special characters and encoding."""
        mock_page = Mock()
        special_text = "Text with special chars: àáâã ñ ç 中文 🙂"
        mock_page.extract_text.return_value = special_text
        mock_pdf_reader_instance = Mock()
        mock_pdf_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_pdf_reader_instance

        with patch('pathlib.Path.exists', return_value=True):
            with patch('audify.readers.pdf.clean_text', return_value="cleaned"):
                reader = PdfReader(temp_pdf_path)
                result = reader.read()

        assert result == special_text

    @patch('builtins.open', new_callable=mock_open)
    def test_save_cleaned_text_with_path_object(self, mock_file_open, temp_pdf_path):
        """Test saving cleaned text to file using Path object."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('audify.readers.pdf.PyPDF2.PdfReader') as mock_pdf_reader:
                with patch(
                    'audify.readers.pdf.clean_text',
                    return_value="Cleaned test content"):
                    # Setup basic mocks for initialization
                    mock_page = Mock()
                    mock_page.extract_text.return_value = "raw content"
                    mock_pdf_reader_instance = Mock()
                    mock_pdf_reader_instance.pages = [mock_page]
                    mock_pdf_reader.return_value = mock_pdf_reader_instance

                    reader = PdfReader(temp_pdf_path)
                    output_path = Path("output.txt")

                    reader.save_cleaned_text(output_path)

        mock_file_open.assert_called_with(output_path, "w", encoding="utf-8")
        mock_file_open().write.assert_called_with("Cleaned test content")

    @patch('builtins.open', new_callable=mock_open)
    def test_save_cleaned_text_with_string(self, mock_file_open, temp_pdf_path):
        """Test saving cleaned text to file using string filename."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('audify.readers.pdf.PyPDF2.PdfReader') as mock_pdf_reader:
                with patch(
                    'audify.readers.pdf.clean_text',
                    return_value="Cleaned test content"):
                    # Setup basic mocks for initialization
                    mock_page = Mock()
                    mock_page.extract_text.return_value = "raw content"
                    mock_pdf_reader_instance = Mock()
                    mock_pdf_reader_instance.pages = [mock_page]
                    mock_pdf_reader.return_value = mock_pdf_reader_instance

                    reader = PdfReader(temp_pdf_path)
                    output_filename = "output.txt"

                    reader.save_cleaned_text(output_filename)

        mock_file_open.assert_called_with(output_filename, "w", encoding="utf-8")
        mock_file_open().write.assert_called_with("Cleaned test content")

    @patch('audify.readers.pdf.PyPDF2.PdfReader')
    @patch('builtins.open', new_callable=mock_open)
    def test_pypdf2_exception_handling(
        self, mock_file_open, mock_pdf_reader, temp_pdf_path):
        """Test handling of PyPDF2 exceptions during PDF reading."""
        mock_pdf_reader.side_effect = PyPDF2.errors.PdfReadError("Corrupted PDF file")

        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(PyPDF2.errors.PdfReadError):
                PdfReader(temp_pdf_path)

    @patch('audify.readers.pdf.clean_text')
    @patch('audify.readers.pdf.PyPDF2.PdfReader')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_text_cleaning_integration(
        self, mock_file_open, mock_exists,
        mock_pdf_reader, mock_clean_text, temp_pdf_path):
        """Test integration with text cleaning utility."""
        mock_exists.return_value = True

        # Setup PDF content with text that needs cleaning
        raw_text = "Raw   text  with\n\nextra   spaces\t\tand\ttabs"
        cleaned_text = "Raw text with extra spaces and tabs"

        mock_page = Mock()
        mock_page.extract_text.return_value = raw_text
        mock_pdf_reader_instance = Mock()
        mock_pdf_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_clean_text.return_value = cleaned_text

        reader = PdfReader(temp_pdf_path)

        # Verify clean_text was called with raw extracted text
        mock_clean_text.assert_called_once_with(raw_text)
        assert reader.text == raw_text
        assert reader.cleaned_text == cleaned_text

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_file_open_exception(self, mock_file_open, temp_pdf_path):
        """Test handling of file I/O exceptions."""
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(IOError, match="Permission denied"):
                PdfReader(temp_pdf_path)

    @patch('audify.readers.pdf.clean_text')
    @patch('audify.readers.pdf.PyPDF2.PdfReader')
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_path_resolution(
        self, mock_file_open, mock_exists, mock_pdf_reader, mock_clean_text
        ):
        """Test that paths are properly resolved to absolute paths."""
        mock_exists.return_value = True
        mock_page = Mock()
        mock_page.extract_text.return_value = "content"
        mock_pdf_reader_instance = Mock()
        mock_pdf_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_clean_text.return_value = "cleaned content"

        # Test with relative path
        relative_path = "relative/path/file.pdf"
        reader = PdfReader(relative_path)

        # Verify path was resolved to absolute
        assert reader.path.is_absolute()
        assert reader.path == Path(relative_path).resolve()
