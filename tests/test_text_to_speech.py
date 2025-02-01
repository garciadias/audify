from pathlib import Path
from tempfile import TemporaryDirectory
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

    return EpubSynthesizer(path="test.epub")


def test_sentence_to_speech(synthesizer):
    sentence = "This is a test sentence."
    synthesizer.sentence_to_speech(sentence)
    synthesizer.model.tts_to_file.assert_called()


@patch("audify.text_to_speech.AudioSegment")
def test_synthesize_chapter(MockAudioSegment, synthesizer):
    MockAudioSegment.from_wav.return_value
    synthesizer.synthesize_chapter("chapter1", 1, "audiobook_path", "en")
    synthesizer.model.tts_to_file.assert_called()
    MockAudioSegment.from_wav.assert_called()


@patch("audify.text_to_speech.subprocess.run")
@patch("audify.text_to_speech.AudioSegment")
def test_create_m4b(MockAudioSegment, mock_subprocess_run, synthesizer):
    mock_audio_segment = MockAudioSegment.from_wav.return_value
    mock_audio_segment.export = MagicMock()
    synthesizer.cover_image = None
    with patch("audify.text_to_speech.Path.unlink"):
        synthesizer.create_m4b()
    mock_subprocess_run.assert_called()


def test_log_on_chapter_file(synthesizer):
    with TemporaryDirectory() as tmp_dir:
        chapter_file_path = Path(tmp_dir) / "chapter.txt"
        start = 0
        duration = 10.0
        title = "Test Chapter"
        end = synthesizer.log_on_chapter_file(chapter_file_path, title, start, duration)
        assert end == start + int(duration * 1000)
        with open(chapter_file_path.parent / "chapters.txt", "r") as f:
            content = f.read()
            assert "TITLE=Test Chapter" in content


@patch("audify.text_to_speech.get_wav_duration")
def test_process_chapter(mock_get_wav_duration, synthesizer):
    mock_get_wav_duration.return_value = 10.0
    chapter_start = synthesizer.process_chapter(1, "chapter1", 0)
    assert chapter_start == 0


@patch("audify.text_to_speech.get_wav_duration")
def test_process_chapters(mock_get_wav_duration, synthesizer):
    mock_get_wav_duration.return_value = 10.0
    synthesizer.process_chapters()
    assert synthesizer.reader.get_chapters.call_count == 1


@patch("audify.text_to_speech.get_wav_duration")
@patch("audify.text_to_speech.EpubSynthesizer.create_m4b")
def test_synthesize(mock_create_m4b, mock_get_wav_duration, synthesizer):
    mock_get_wav_duration.return_value = 10.0
    result = synthesizer.synthesize()
    assert "test_title" in result
    mock_create_m4b.assert_called_once()
