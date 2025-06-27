from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydub.exceptions import CouldntDecodeError

from audify.utils import (break_text_into_sentences, break_too_long_sentences,
                          clean_text, combine_small_sentences,
                          get_audio_duration, get_file_extension,
                          get_file_name_title, sentence_to_speech)

MODULE_PATH = Path(__file__).resolve().parents[1]


@pytest.fixture
def pdf_file_create(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("This is a test PDF file.")
    return pdf_path


def test_text_is_cleaned(readers):
    for reader in readers:
        cleaned_text = clean_text(reader.get_cleaned_text())
        assert cleaned_text
        assert ".." not in cleaned_text
        assert "  " not in cleaned_text
        assert "\n" not in cleaned_text
        assert "//" not in cleaned_text


def test_clean_text():
    text = " This is a test text.  With  multiple spaces, tabs\t, and newlines\n."
    expected = "This is a test text. With multiple spaces, tabs, and newlines."
    assert clean_text(text) == expected


def test_combine_small_sentences():
    sentences = ["This is a test.", "Short.", "Another test sentence."]
    expected = ["This is a test. Short.", "Another test sentence."]
    assert combine_small_sentences(sentences, min_length=10) == expected


def test_break_too_long_sentences():
    sentences = [
        "This is a very long sentence that needs to be broken into smaller parts."
    ]
    expected = [
        "This is a very long sentence that needs to be",
        "broken into smaller parts.",
    ]
    assert break_too_long_sentences(sentences, max_length=50) == expected


def test_break_text_into_sentences():
    text = "This is a test. This is another test sentence."
    expected = ["This is a test.", "This is another test sentence."]
    assert break_text_into_sentences(text, max_length=50, min_length=10) == expected


@patch("audify.utils.AudioSegment.from_file")
def test_get_audio_duration(mock_from_file):
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = 10000  # 10 seconds
    mock_from_file.return_value = mock_audio
    assert get_audio_duration("test.wav") == 10.0


def test_get_audio_duration_culdnt_decode_error():
    with patch("audify.utils.AudioSegment.from_file", side_effect=CouldntDecodeError):
        assert get_audio_duration("test.wav") == 0.0


@patch("audify.utils.TTS")
@patch("audify.utils.Path")
def test_sentence_to_speech(mock_path, mock_tts):
    mock_model = MagicMock()
    mock_path.return_value = Path("/tmp/speech.wav")
    sentence_to_speech("This is a test sentence.", mock_model)
    mock_model.tts_to_file.assert_called()


@patch("audify.utils.TTS")
@patch("audify.utils.Path")
def test_sentence_to_speech_not_multilingual(mock_path, mock_tts):
    mock_model = MagicMock()
    mock_model.is_multi_lingual = False
    mock_path.return_value = Path("/tmp/speech.wav")
    sentence_to_speech("This is a test sentence.", mock_model)
    mock_model.tts_to_file.assert_called()


def test_get_file_extension():
    assert get_file_extension("test.wav") == ".wav"


def test_get_file_name_title():
    title = "This is a Test Title!"
    expected = "this_is_a_test_title"
    assert get_file_name_title(title) == expected


@patch("audify.utils.TTS")
@patch("audify.utils.Path")
def test_sentence_to_speech_multilingual(mock_path, mock_tts):
    mock_model = MagicMock()
    mock_model.is_multi_lingual = True
    mock_path.return_value = Path("/tmp/speech.wav")
    sentence_to_speech("This is a test sentence.", mock_model)
    mock_model.tts_to_file.assert_called()


@patch("audify.utils.TTS")
@patch("audify.utils.Path")
@pytest.mark.parametrize(
    'multilingual', [True, False]
)
def test_sentence_to_speech_read_error(mock_path, mock_tts, multilingual):
    mock_model = MagicMock()
    mock_model.is_multi_lingual = multilingual
    mock_model.tts_to_file.side_effect = [KeyError("Test KeyError"), True]
    mock_path.return_value = Path("/tmp/")
    sentence_to_speech("This is a test sentence.", mock_model)
    if multilingual:
        mock_model.tts_to_file.assert_called_with(
            text="Error: 'Test KeyError'",
            file_path=mock_path.return_value / "speech.wav",
            language="en",
            speaker_wav="data/Jennifer_16khz.wav",
            speed=1.15,
        )
    else:
        mock_model.tts_to_file.assert_called_with(
            text="Error: 'Test KeyError'",
            file_path=mock_path.return_value / "speech.wav",
            speaker_wav="data/Jennifer_16khz.wav",
            speed=1.15,
        )


@patch("audify.utils.TTS")
@patch("audify.utils.Path")
def test_sentence_to_speech_exception(mock_path, mock_tts):
    mock_model = MagicMock()
    mock_model.tts_to_file.side_effect = KeyError("Test KeyError")
    mock_path.return_value = Path("/tmp/speech.wav")
    with pytest.raises(KeyError):
        sentence_to_speech("This is a test sentence.", mock_model)
    assert mock_model.tts_to_file.call_count == 2


@patch("audify.utils.TTS")
@patch("audify.utils.Path")
def test_sentence_to_speech_output_dir_str(mock_path, mock_tts):
    mock_model = MagicMock()
    mock_path.return_value = Path("/tmp/test/speech.wav")
    sentence_to_speech("This is a test sentence.", mock_model, output_dir="/tmp/test")
    mock_model.tts_to_file.assert_called()


@patch("audify.utils.TTS")
@patch("audify.utils.Path")
def test_sentence_to_speech_output_dir_path(mock_path, mock_tts):
    mock_model = MagicMock()
    output_dir = Path("/tmp/test")
    mock_path.return_value = output_dir / "speech.wav"
    sentence_to_speech("This is a test sentence.", mock_model, output_dir=output_dir)
    mock_model.tts_to_file.assert_called()


@patch("audify.utils.TTS")
@patch("audify.utils.Path")
def test_sentence_to_speech_directory_creation(mock_path, mock_tts):
    mock_model = MagicMock()
    mock_path.return_value = MagicMock()
    mock_path.return_value.parent.is_dir.return_value = False
    sentence_to_speech("This is a test sentence.", mock_model, output_dir="/tmp/nonexistent/speech.wav")
    mock_path("/tmp/nonexistent/speech.wav").parent.mkdir.assert_called_with(parents=True, exist_ok=True)


@patch("audify.utils.TTS")
@patch("audify.utils.Path")
def test_sentence_to_speech_no_directory_creation(mock_path, mock_tts):
    mock_model = MagicMock()
    mock_path.return_value = MagicMock()
    mock_path.parent.is_dir.return_value = True
    sentence_to_speech("This is a test sentence.", mock_model, output_dir="/tmp/speech.wav")
    mock_path("/tmp/speech.wav").parent.mkdir.assert_not_called()
