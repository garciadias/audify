# tests/test_start.py
import sys
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Mock the audify modules before importing start
# Keep mocks for underlying synthesizers and utils
mock_epub_synthesizer = MagicMock()
mock_pdf_synthesizer = MagicMock()
mock_inspect_synthesizer = MagicMock()

# Mock the classes themselves
mock_epub_synthesizer_class = MagicMock(return_value=mock_epub_synthesizer)
mock_pdf_synthesizer_class = MagicMock(return_value=mock_pdf_synthesizer)
mock_inspect_synthesizer_class = MagicMock(return_value=mock_inspect_synthesizer)

# Mock the utility function
mock_get_file_extension = MagicMock()


# Apply patches using a dictionary
patches = {
    "audify.text_to_speech.EpubSynthesizer": mock_epub_synthesizer_class,
    "audify.text_to_speech.PdfSynthesizer": mock_pdf_synthesizer_class,
    "audify.text_to_speech.InspectSynthesizer": mock_inspect_synthesizer_class,
    "audify.start.get_file_extension": mock_get_file_extension,
    "os.get_terminal_size": MagicMock(return_value=(80, 24)),  # Mock terminal size
}

# Use pytest's patching capabilities or context managers in tests
# Import start *after* potentially mocking dependencies if needed at module level
# For simplicity here, we'll patch within tests.
from audify import start  # noqa: E402


@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance."""
    return CliRunner()


@patch.dict(sys.modules, patches)
@patch("pathlib.Path.exists", return_value=True)  # Mock file existence check
def test_main_epub_synthesis(mock_exists, runner):
    """Test main command with an EPUB file."""
    mock_get_file_extension.return_value = ".epub"
    result = runner.invoke(
        start.main,
        [
            "fake.epub",
            "--language",
            "en",
            "--voice",
            "voice.wav",
            "--model",
            "model_name",
            "-y",
        ],
    )

    assert result.exit_code == 2
    mock_pdf_synthesizer_class.assert_not_called()
    mock_inspect_synthesizer_class.assert_not_called()


@patch.dict(sys.modules, patches)
@patch("pathlib.Path.exists", return_value=True)  # Mock file existence check
def test_main_epub_synthesis_abort_confirmation(mock_exists, runner):
    """Test main command with an EPUB file aborting confirmation."""
    mock_get_file_extension.return_value = ".epub"
    # Simulate user typing 'n' then Enter for confirmation
    result = runner.invoke(start.main, ["fake.epub"], input="n\n")
    assert result.exit_code == 2
