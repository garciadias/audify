from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audify.text_to_speech import EpubSynthesizer, InspectSynthesizer, PdfSynthesizer
from audify.utils import sentence_to_speech


@patch("audify.text_to_speech.TTS")
@patch("audify.text_to_speech.EpubReader.epub.read_epub")
@pytest.fixture
def epub_synthesizer():
    return EpubSynthesizer(path="test.epub")


@pytest.fixture
def pdf_synthesizer():
    return PdfSynthesizer(pdf_path="test.pdf")


@pytest.fixture
def inspect_synthesizer():
    return InspectSynthesizer()


@patch("audify.text_to_speech.TTS")
@patch("audify.text_to_speech.Path")
def test_sentence_to_speech(mock_path, mock_tts):
    mock_model = MagicMock()
    mock_path.return_value = Path("/tmp/speech.wav")
    sentence_to_speech("This is a test sentence.", mock_model)
    mock_model.tts_to_file.assert_called()


@pytest.mark.skip(
    reason="The default model changed from tts to kokoro, "
    "this test needs to be updated."
)
@patch("audify.text_to_speech.AudioSegment")
def test_synthesize_chapter(MockAudioSegment, synthesizer):
    MockAudioSegment.from_wav.return_value
    synthesizer.synthesize_chapter("chapter1", 1)
    synthesizer.model.tts_to_file.assert_called()
    MockAudioSegment.from_wav.assert_called()


def test_log_on_chapter_file(synthesizer):
    start = 0
    duration = 10.0
    title = "Test Chapter"
    end = synthesizer._log_chapter_metadata(title, start, duration)
    assert end == start + int(duration * 1000)


@patch("audify.text_to_speech.get_audio_duration")
def test_process_chapter(mock_get_audio_duration, synthesizer):
    mock_get_audio_duration.return_value = 10.0
    chapter_start = synthesizer._process_single_chapter(1, "chapter1", 0)
    assert chapter_start == 0


@patch("audify.text_to_speech.get_audio_duration")
@patch("audify.text_to_speech.EpubSynthesizer.create_m4b")
@patch("audify.text_to_speech.input", return_value="y")
def test_synthesize(mock_get_audio_duration, synthesizer, mock_input):
    del mock_input
    mock_get_audio_duration.return_value = 10.0
    synthesizer.synthesize()


if __name__ == "__main__":
    pytest.main()
