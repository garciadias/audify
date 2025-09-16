# tests/test_start.py
import sys
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from reportlab.pdfgen import canvas

from audify import start

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


@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance."""
    return CliRunner()


@patch.dict(sys.modules, patches)
@patch("pathlib.Path.exists", return_value=True)  # Mock file existence check
def test_main_epub_synthesis(mock_exists, runner):
    """Test main command with an EPUB file."""
    mock_get_file_extension.return_value = ".epub"
    with TemporaryDirectory() as temp_dir:
        # Create a temporary EPUB file
        epub_file_path = f"{temp_dir}/test.epub"
        with open(epub_file_path, "w") as f:
            f.write("Fake EPUB content")

            result = runner.invoke(
                start.main,
                [
                    epub_file_path,
                    "--language",
                    "en",
                    "--voice",
                    "voice.wav",
                    "--model",
                    "model_name",
                    "--engine",
                    "tts_models",
                    "--save-text",
                    "--translate",
                ],
            )
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
def test_main_epub_synthesis_abort_confirmation(mock_get_file_extension, runner):
    """Test main command with an EPUB file aborting confirmation."""
    mock_get_file_extension.return_value = ".epub"
    # Simulate user typing 'n' then Enter for confirmation
    result = runner.invoke(start.main, ["fake.epub"], input="n\n")
    assert result.exit_code == 2

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
    # We need to apply patches before importing start if mocks affect import time
    # Using patch.dict on sys.modules handles this if done before import.
    # However, the example structure imports start first, then patches in tests.
    # Let's assume the mocks are intended to be active during test execution only.
    with patch.dict(sys.modules, patches):
        pass  # noqa: E402


def reset_mocks_fixture():
    """Reset mocks before each test function."""
    mock_epub_synthesizer_class.reset_mock()
    mock_pdf_synthesizer_class.reset_mock()
    mock_inspect_synthesizer_class.reset_mock()
    mock_get_file_extension.reset_mock()
    mock_epub_synthesizer.reset_mock()
    mock_pdf_synthesizer.reset_mock()
    mock_inspect_synthesizer.reset_mock()
    # Reset mock model attributes if they were set directly
    if hasattr(mock_inspect_synthesizer, "model"):
        # Ensure the attribute exists before trying to delete it
        try:
            del mock_inspect_synthesizer.model
        except AttributeError:
            pass  # Already deleted or never set


@patch("os.get_terminal_size", return_value=(80, 24))  # Mock terminal size
def test_main_list_languages(mock_exists, runner):
    """Test main command with --list-languages flag."""
    # Setup mock model attribute for languages
    result = runner.invoke(start.main, ["--list-languages"])

    assert result.exit_code == 0


@patch("pathlib.Path.exists", return_value=True)
@patch("os.get_terminal_size", return_value=(80, 24))  # Mock terminal size
def test_main_list_models(mock_exists, mock_terminal_size, runner):
    """Test main command with --list-models flag."""
    # Setup mock model attribute for models
    result = runner.invoke(start.main, ["--list-models"])

    assert result.exit_code == 0


@patch("pathlib.Path.exists", return_value=True)
@patch("os.get_terminal_size", return_value=(80, 24))
def test_main_pdf_synthesis(mock_exists, mock_terminal_size, runner):
    """Test main command with a PDF file."""
    with TemporaryDirectory() as temp_dir:
        # Create a temporary PDF file
        pdf_file_path = f"{temp_dir}/test.pdf"
        # Use reportlab to create a PDF with text content

        c = canvas.Canvas(pdf_file_path, pagesize=(8.27 * 72, 11.7 * 72))
        c.drawString(100, 700, "This is a test PDF content.")
        c.save()
        mock_get_file_extension.return_value = True

        # Invoke the main command but mock the PDF synthesizer
        mock_pdf_synthesizer_class.return_value = mock_pdf_synthesizer
        mock_pdf_synthesizer.synthesize.return_value = None  # Mock synthesize method
        mock_get_file_extension.return_value = ".pdf"
        # Now invoke the main command with the PDF file mocking the synthesizer run
        module_path = "audify.text_to_speech"
        with patch(f"{module_path}.requests.post") as mock_synthesize_kokoro:
            mock_synthesize_kokoro.return_value = MagicMock()
            mock_synthesize_kokoro.return_value.json.return_value = {
                "data": {"audio_url": "http://example.com/audio.mp3"}
            }
            result = runner.invoke(
                start.main,
                [
                    pdf_file_path,
                    "--language",
                    "fr",
                    "--voice",
                    "af_bella",
                    "--save-text",
                    "--translate",
                    "en",
                ],
            )

        assert result.exit_code == 0
        assert "PDF to mp3" in result.output
