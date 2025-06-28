from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audify.text_to_speech import (
    BaseSynthesizer,
    EpubSynthesizer,
    InspectSynthesizer,
    PdfSynthesizer,
)
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


@patch("audify.text_to_speech.TTS")
def test_inspect_synthesizer_init(mock_tts):
    synthesizer = InspectSynthesizer()
    mock_tts.assert_called_with(model_name=InspectSynthesizer.DEFAULT_MODEL)
    assert synthesizer.model_name == InspectSynthesizer.DEFAULT_MODEL


@patch("audify.text_to_speech.TTS")
def test_inspect_synthesizer_init_with_model_name(mock_tts):
    model_name = "test_model"
    synthesizer = InspectSynthesizer(model_name=model_name)
    mock_tts.assert_called_with(model_name=model_name)
    assert synthesizer.model_name == model_name


@patch("audify.text_to_speech.TTS")
def test_inspect_synthesizer_init_exception(mock_tts):
    mock_tts.side_effect = Exception("Failed to load model")
    with pytest.raises(Exception, match="Failed to load model"):
        InspectSynthesizer()


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


@patch("audify.text_to_speech.AudioSegment")
@patch("audify.text_to_speech.tqdm.tqdm")
@patch("audify.text_to_speech.suppress_stdout")
def test_synthesize_tts_models_success(
    mock_suppress_stdout, mock_tqdm, mock_audio_segment, base_synthesizer
):
    sentences = ["sentence 1", "sentence 2"]
    output_wav_path = Path("/tmp/output.wav")
    mock_tqdm.return_value = sentences
    mock_audio_segment.from_wav.return_value = MagicMock()
    base_synthesizer.model.tts_to_file.return_value = None
    temp_speech_path = base_synthesizer.tmp_dir / "speech_segment.wav"
    temp_speech_path.touch()

    BaseSynthesizer._synthesize_tts_models(base_synthesizer, sentences, output_wav_path)

    assert base_synthesizer.model.tts_to_file.call_count == len(sentences)
    assert mock_audio_segment.from_wav.call_count == 1
    assert Path(temp_speech_path).exists() is False


@patch("audify.text_to_speech.AudioSegment")
@patch("audify.text_to_speech.tqdm.tqdm")
@patch("audify.text_to_speech.suppress_stdout")
def test_synthesize_tts_models_empty_sentence(
    mock_suppress_stdout, mock_tqdm, mock_audio_segment, base_synthesizer
):
    sentences = ["", "  "]
    output_wav_path = Path("/tmp/output.wav")
    mock_tqdm.return_value = sentences
    BaseSynthesizer._synthesize_tts_models(base_synthesizer, sentences, output_wav_path)
    assert base_synthesizer.model.tts_to_file.call_count == 0


@patch("audify.text_to_speech.AudioSegment")
@patch("audify.text_to_speech.tqdm.tqdm")
@patch("audify.text_to_speech.suppress_stdout")
def test_synthesize_tts_models_tts_exception(
    mock_suppress_stdout, mock_tqdm, mock_audio_segment, base_synthesizer
):
    sentences = ["sentence 1", "sentence 2"]
    output_wav_path = Path("/tmp/output.wav")
    mock_tqdm.return_value = sentences
    base_synthesizer.model.tts_to_file.side_effect = Exception("TTS Error")
    temp_speech_path = base_synthesizer.tmp_dir / "speech_segment.wav"
    temp_speech_path.touch()

    BaseSynthesizer._synthesize_tts_models(base_synthesizer, sentences, output_wav_path)

    assert base_synthesizer.model.tts_to_file.call_count == len(sentences)
    assert Path(temp_speech_path).exists() is False


@patch("audify.text_to_speech.AudioSegment")
@patch("audify.text_to_speech.tqdm.tqdm")
@patch("audify.text_to_speech.suppress_stdout")
def test_synthesize_tts_models_file_not_found(
    mock_suppress_stdout, mock_tqdm, mock_audio_segment, base_synthesizer
):
    sentences = ["sentence 1", "sentence 2"]
    output_wav_path = Path("/tmp/output.wav")
    mock_tqdm.return_value = sentences
    base_synthesizer.model.tts_to_file.return_value = None
    mock_audio_segment.from_wav.side_effect = FileNotFoundError("File not found")
    temp_speech_path = base_synthesizer.tmp_dir / "speech_segment.wav"
    temp_speech_path.touch()

    BaseSynthesizer._synthesize_tts_models(base_synthesizer, sentences, output_wav_path)

    assert base_synthesizer.model.tts_to_file.call_count == len(sentences)
    assert Path(temp_speech_path).exists() is False


@patch("audify.text_to_speech.AudioSegment")
@patch("audify.text_to_speech.tqdm.tqdm")
@patch("audify.text_to_speech.suppress_stdout")
def test_synthesize_tts_models_no_output_file(
    mock_suppress_stdout, mock_tqdm, mock_audio_segment, base_synthesizer
):
    sentences = ["sentence 1", "sentence 2"]
    output_wav_path = Path("/tmp/output.wav")
    mock_tqdm.return_value = sentences
    base_synthesizer.model.tts_to_file.return_value = None
    temp_speech_path = base_synthesizer.tmp_dir / "speech_segment.wav"

    BaseSynthesizer._synthesize_tts_models(base_synthesizer, sentences, output_wav_path)

    assert base_synthesizer.model.tts_to_file.call_count == len(sentences)
    assert Path(temp_speech_path).exists() is False


if __name__ == "__main__":
    pytest.main()
