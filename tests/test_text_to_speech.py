from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audify.text_to_speech import LOADED_MODEL, TextToSpeech


@pytest.fixture
def synthesizer():
    return TextToSpeech(LOADED_MODEL, "cpu")


def test_sentence_to_speech(synthesizer):
    with patch.object(LOADED_MODEL, "tts_to_file") as mock_tts_to_file:
        synthesizer.sentence_to_speech("Hello world", "tmp/speech.wav")
        mock_tts_to_file.assert_called_once_with(
            text="Hello world",
            file_path="tmp/speech.wav",
            language="es",
            speaker_wav="data/Jennifer_16khz.wav",
        )


def test_synthesize_chapter(synthesizer):
    with (
        patch.object(synthesizer, "sentence_to_speech") as mock_sentence_to_speech,
        patch(
            "pydub.AudioSegment.from_wav", return_value=MagicMock()
        ) as mock_audio_segment,
    ):
        synthesizer.synthesize_chapter("Chapter content", 1, "output", "en")
        assert mock_sentence_to_speech.call_count > 0
        mock_audio_segment.assert_called()


def test_create_m4b(synthesizer):
    with (
        patch(
            "pydub.AudioSegment.from_wav", return_value=MagicMock()
        ) as mock_audio_segment,
        patch("subprocess.run") as mock_subprocess_run,
    ):
        synthesizer.create_m4b(
            ["chapter1.wav", "chapter2.wav"], "output/book.epub", "cover.jpg"
        )
        mock_audio_segment.assert_called()
        mock_subprocess_run.assert_called()


def test_get_wav_duration(synthesizer):
    with patch("wave.open", return_value=MagicMock()) as mock_wave_open:
        mock_wave_open.return_value.getnframes.return_value = 44100
        mock_wave_open.return_value.getframerate.return_value = 44100
        duration = synthesizer.get_wav_duration("test.wav")
        assert duration == 1.0


def test_log_on_chapter_file(synthesizer):
    chapter_file_path = "output/chapter_1.txt"
    title = "Chapter 1"
    start = 0
    duration = 60.0
    end = synthesizer.log_on_chapter_file(chapter_file_path, title, start, duration)
    assert end == 60000
    with open(Path(chapter_file_path).parent / "chapters.txt", "r") as f:
        content = f.read()
    assert "TITLE=Chapter 1" in content


def test_process_chapter(synthesizer):
    with (
        patch.object(synthesizer, "synthesize_chapter") as mock_synthesize_chapter,
        patch.object(
            synthesizer, "get_wav_duration", return_value=60.0
        ) as mock_get_wav_duration,
        patch.object(
            synthesizer, "log_on_chapter_file", return_value=60000
        ) as mock_log_on_chapter_file,
    ):
        chapter_start = synthesizer.process_chapter(
            1, "Chapter content", 0, "output", "en"
        )
        assert chapter_start == 60000
        mock_synthesize_chapter.assert_called_once()
        mock_get_wav_duration.assert_called_once()
        mock_log_on_chapter_file.assert_called_once()
