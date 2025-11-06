from unittest.mock import MagicMock, patch

import pytest

from audify.text_to_speech import EpubSynthesizer


@pytest.fixture
@patch("audify.text_to_speech.EpubReader")
@patch("audify.text_to_speech.TTS")
def synthesizer(MockTTS, MockEpubReader):
    mock_reader = MockEpubReader.return_value
    mock_reader.get_language.return_value = "en"
    mock_reader.title = "test_title"
    mock_reader.get_cover_image.return_value = None
    mock_reader.get_chapters.return_value = ["chapter1", "chapter2"]
    mock_reader.extract_text.return_value = "This is a test sentence."
    mock_reader.break_text_into_sentences.return_value = ["This is a test sentence."]
    mock_reader.get_chapter_title.return_value = "Test Chapter"
    mock_tts = MockTTS.return_value
    mock_tts.tts_to_file = MagicMock()

    return EpubSynthesizer(path="test.epub", engine="tts_models")
