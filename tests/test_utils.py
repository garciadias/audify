from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from audify.utils import break_text_into_sentences, clean_text, sentence_to_speech
from tests.fixtures.readers import READERS

MODULE_PATH = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("reader", READERS)
def test_text_is_cleaned(reader):
    cleaned_text = clean_text(reader.get_cleaned_text())
    assert cleaned_text
    assert ".." not in cleaned_text
    assert "  " not in cleaned_text
    assert "\n" not in cleaned_text
    assert "//" not in cleaned_text


@pytest.mark.parametrize("reader", READERS)
def test_break_text_into_sentences(reader):
    max_length = 239
    min_length = 10
    sentences = break_text_into_sentences(
        reader.get_cleaned_text(), max_length, min_length
    )
    assert sentences
    # Check that all sentences are shorter than 239 characters
    assert all([len(sentence) < max_length for sentence in sentences])
    # Check that all sentences are longer than 10 characters
    assert all([len(sentence) > min_length for sentence in sentences])


def test_sentence_to_speech(synthesizer):
    sentence = "This is a test sentence."
    model = synthesizer.model
    tmp_file = NamedTemporaryFile(suffix=".wav")
    sentence_to_speech(sentence, model, tmp_file.name)
    assert Path(tmp_file.name).exists()
    tmp_file.close()
    assert not Path(tmp_file.name).exists()
