from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audify.text_to_speech import EpubSynthesizer, PdfSynthesizer


@patch("audify.text_to_speech.TTS")
@patch("audify.text_to_speech.EpubReader.epub.read_epub")
@pytest.fixture
def epub_synthesizer():
    return EpubSynthesizer(path="test.epub")


@pytest.fixture
def pdf_synthesizer():
    return PdfSynthesizer(pdf_path="test.pdf")


@patch("audify.text_to_speech.get_audio_duration")
@patch("audify.text_to_speech.EpubSynthesizer.create_m4b")
@patch("audify.text_to_speech.input", return_value="y")
def test_synthesize(mock_get_audio_duration, synthesizer, mock_input):
    del mock_input
    mock_get_audio_duration.return_value = 10.0
    synthesizer.synthesize()


@patch("audify.text_to_speech.EpubReader")
def test_epub_synthesizer_init(mock_epub_reader):
    mock_epub_reader_instance = MagicMock()
    mock_epub_reader.return_value = mock_epub_reader_instance
    mock_epub_reader_instance.get_language.return_value = "en"
    mock_epub_reader_instance.title = "Test Title"

    synthesizer = EpubSynthesizer(path="test.epub")

    assert synthesizer.language == "en"
    assert synthesizer.title == "Test Title"
    assert synthesizer.file_name == "test_title"


@patch("audify.text_to_speech.EpubReader")
def test_epub_synthesizer_init_no_language(mock_epub_reader):
    mock_epub_reader_instance = MagicMock()
    mock_epub_reader.return_value = mock_epub_reader_instance
    mock_epub_reader_instance.get_language.return_value = None
    with pytest.raises(ValueError, match="Language must be provided"):
        EpubSynthesizer(path="test.epub")


@patch("audify.text_to_speech.PdfReader")
def test_pdf_synthesizer_init(mock_pdf_reader):
    pdf_path = "test.pdf"
    with patch("audify.text_to_speech.Path.exists", return_value=True):
        mock_pdf_reader_instance = MagicMock()
        mock_pdf_reader.return_value = mock_pdf_reader_instance
        mock_pdf_reader_instance.get_language.return_value = "en"
        synthesizer = PdfSynthesizer(pdf_path=pdf_path)
    assert synthesizer.path == Path(pdf_path).resolve()
    assert synthesizer.language == "en"


def test_pdf_synthesizer_init_file_not_found():
    pdf_path = "nonexistent.pdf"
    with pytest.raises(
        FileNotFoundError, match=f"PDF file not found at {Path(pdf_path).resolve()}"
    ):
        PdfSynthesizer(pdf_path=pdf_path)


@pytest.fixture
def base_synthesizer():
    synthesizer = MagicMock()
    synthesizer.model = MagicMock()
    synthesizer.tmp_dir = Path("/tmp")
    return synthesizer


if __name__ == "__main__":
    pytest.main()
