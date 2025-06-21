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
    assert "Available languages:".center(80) in result.output
    assert "en, es, fr" in result.output


@patch("pathlib.Path.exists", return_value=True)
@patch("os.get_terminal_size", return_value=(80, 24))  # Mock terminal size
def test_main_list_models(mock_exists, mock_terminal_size, runner):
    """Test main command with --list-models flag."""
    # Setup mock model attribute for models
    result = runner.invoke(start.main, ["--list-models"])

    assert result.exit_code == 0
    assert "Available models:" in result.output
    assert "tts_models" in result.output


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
        with patch(f"{module_path}.KPipeline") as mock_synthesize_kokoro:
            mock_synthesize_kokoro.return_value = MagicMock()
            mock_synthesize_kokoro.return_value.pipeline = MagicMock()
            mock_synthesize_kokoro.return_value.pipeline.run.return_value = None
            result = runner.invoke(
                start.main,
                [
                    pdf_file_path,
                    "--language",
                    "fr",
                    "--voice",
                    "speaker.wav",
                    "--save-text",
                    "--translate",
                    "en",
                ],
            )

        assert result.exit_code == 0
        assert "PDF to mp3" in result.output


@pytest.mark.skip
@patch.dict(sys.modules, patches)
@patch("pathlib.Path.exists", return_value=True)
def test_main_unsupported_format(mock_exists, runner):
    """Test main command with an unsupported file format."""
    mock_get_file_extension.return_value = ".txt"
    result = runner.invoke(start.main, ["fake.txt"])

    # Expecting click to catch the ValueError and exit non-zero
    assert result.exit_code != 0
    # Check if the exception message is printed (Click might wrap it)
    assert "Unsupported file format" in result.output
    mock_get_file_extension.assert_called_once_with("fake.txt")
    mock_pdf_synthesizer_class.assert_not_called()
    mock_epub_synthesizer_class.assert_not_called()
    mock_inspect_synthesizer_class.assert_not_called()


# --- Refined EPUB Tests ---


@pytest.mark.skip
@patch.dict(sys.modules, patches)
@patch("pathlib.Path.exists", return_value=True)  # Mock file existence check
def test_main_epub_synthesis_with_options_and_skip_confirm(mock_exists, runner):
    """Test main command with an EPUB file, specific options, and -y flag."""
    mock_get_file_extension.return_value = ".epub"
    result = runner.invoke(
        start.main,
        [
            "fake.epub",
            "--language",
            "es",
            "--voice",
            "spanish_voice.wav",
            "--model",
            "spanish_model",
            "--engine",
            "tts_models",
            "--save-text",
            "--translate",
            "en",
            "-y",  # Skip confirmation -> confirm=False
        ],
    )

    assert result.exit_code == 0  # Assuming success if synthesize doesn't raise
    assert "Epub to Audiobook".center(80) in result.output
    mock_get_file_extension.assert_called_once_with("fake.epub")
    mock_epub_synthesizer_class.assert_called_once_with(
        "fake.epub",
        language="es",
        speaker="spanish_voice.wav",
        model_name="spanish_model",
        translate="en",
        save_text=True,
        engine="tts_models",
        confirm=False,  # Because -y is passed
    )
    mock_epub_synthesizer.synthesize.assert_called_once()
    mock_pdf_synthesizer_class.assert_not_called()
    mock_inspect_synthesizer_class.assert_not_called()


@pytest.mark.skip
@patch.dict(sys.modules, patches)
@patch("pathlib.Path.exists", return_value=True)
def test_main_epub_synthesis_confirm_yes(mock_exists, runner):
    """Test main command with EPUB, default options, and confirming 'y'."""
    mock_get_file_extension.return_value = ".epub"
    # Simulate user typing 'y' then Enter for confirmation
    # (default behavior without -y)
    result = runner.invoke(start.main, ["another.epub"], input="y\n")

    assert result.exit_code == 0  # Assuming success if synthesize doesn't raise
    assert "Epub to Audiobook".center(80) in result.output
    mock_get_file_extension.assert_called_once_with("another.epub")
    mock_epub_synthesizer_class.assert_called_once_with(
        "another.epub",
        language="en",  # Default
        speaker="data/Jennifer_16khz.wav",  # Default
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",  # Default
        translate=None,  # Default
        save_text=False,  # Default
        engine="kokoro",  # Default
        confirm=True,  # Default (no -y)
    )
    # Assuming confirmation logic is handled within EpubSynthesizer
    # and 'y' allows synthesize to proceed.
    mock_epub_synthesizer.synthesize.assert_called_once()
    mock_pdf_synthesizer_class.assert_not_called()
    mock_inspect_synthesizer_class.assert_not_called()


@pytest.mark.skip
@patch.dict(sys.modules, patches)
@patch("pathlib.Path.exists", return_value=True)
def test_main_epub_synthesis_confirm_no(mock_exists, runner):
    """Test main command with EPUB, default options, and confirming 'n'."""
    mock_get_file_extension.return_value = ".epub"
    # Simulate user typing 'n' then Enter for confirmation
    # This renames the original test 'test_main_epub_synthesis_abort_confirmation'
    result = runner.invoke(start.main, ["abort.epub"], input="n\n")

    # The original test asserted exit code 2. This implies the confirmation
    # logic (likely within EpubSynthesizer) causes an abort/exit.
    # We keep this assertion based on the previous test's expectation.
    assert result.exit_code == 2
    assert "Epub to Audiobook".center(80) in result.output
    mock_get_file_extension.assert_called_once_with("abort.epub")
    mock_epub_synthesizer_class.assert_called_once_with(
        "abort.epub",
        language="en",
        speaker="data/Jennifer_16khz.wav",
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        translate=None,
        save_text=False,
        engine="kokoro",
        confirm=True,  # confirm=True passed, but user input 'n' causes abort
    )
    # If confirmation ('n') causes an abort *before* or *during* synthesis,
    # synthesize might not be called or might not complete.
    # Depending on how EpubSynthesizer handles 'n' with confirm=True.
    # Let's assume synthesize is called, but internally aborts based on input 'n'.
    # Or perhaps the constructor itself aborts. If constructor aborts, synthesize
    # isn't called. Given exit code 2, it's likely an explicit abort happens.
    # Let's assume synthesize *is* called, matching the mock structure,
    # but the application exits due to the 'n' input being processed.
    mock_epub_synthesizer.synthesize.assert_called_once()
    mock_pdf_synthesizer_class.assert_not_called()
    mock_inspect_synthesizer_class.assert_not_called()
