from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


@patch("audify.text_to_speech.AudioSegment")
def test_synthesize_chapter(MockAudioSegment, synthesizer):
    MockAudioSegment.from_wav.return_value
    synthesizer.synthesize_chapter("chapter1", 1, "audiobook_path")
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


@patch("audify.text_to_speech.get_wav_duration")
def test_process_chapter(mock_get_wav_duration, synthesizer):
    mock_get_wav_duration.return_value = 10.0
    chapter_start = synthesizer.process_chapter(1, "chapter1", 0)
    assert chapter_start == 0


@patch("audify.text_to_speech.get_wav_duration")
@patch("audify.text_to_speech.EpubSynthesizer.create_m4b")
@patch("audify.text_to_speech.input", return_value="y")
def test_synthesize(mock_get_wav_duration, synthesizer, mock_input):
    del mock_input
    mock_get_wav_duration.return_value = 10.0
    synthesizer.synthesize()
