from unittest.mock import MagicMock, patch

import pytest

from audify.translate import translate_sentence


@patch("audify.translate.M2MCG.from_pretrained")
@patch("audify.translate.M2MT.from_pretrained")
def test_translate_sentence(mock_tokenizer_class, mock_model_class):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model_class.return_value = mock_model
    mock_tokenizer_class.return_value = mock_tokenizer

    mock_tokenizer.encode.return_value = {"input_ids": [0]}
    mock_tokenizer.batch_decode.return_value = ["translated sentence"]
    mock_tokenizer.get_lang_id.return_value = 0

    sentence = "This is a test sentence."
    src_lang = "en"
    tgt_lang = "fr"

    translated_sentence = translate_sentence(
        sentence, src_lang=src_lang, tgt_lang=tgt_lang
    )

    assert translated_sentence == "C’est une phrase de test."


if __name__ == "__main__":
    pytest.main()
