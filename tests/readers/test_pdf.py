import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pypdf
import pytest

from audify.readers.pdf import PdfReader


@pytest.fixture
def temp_pdf_path():
    """Create a temporary PDF file path."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_pdf_content():
    """Mock PDF content for testing."""
    return (
        "This is sample PDF text content.\n"
        "With multiple lines.\nAnd some special characters: àáâã"
    )


@pytest.fixture
def mock_cleaned_text():
    """Mock cleaned text output."""
    return (
        "This is sample PDF text content. With multiple lines."
        " And some special characters: àáâã"
    )


class TestPdfReader:
    """Test suite for PdfReader class."""


    @patch("pathlib.Path.exists")
    def test_init_file_not_found(self, mock_exists):
        """Test FileNotFoundError when PDF file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="PDF file not found at"):
            PdfReader("/nonexistent/path/file.pdf")
