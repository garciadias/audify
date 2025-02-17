import pytest

from audify.translate import translate_sentence

TEXTS = {
    "pt": "A vida é como uma caixa de chocolate.",
    "es": "La vida es como una caja de chocolate.",
    "en": "Life is like a box of chocolate.",
    "zh": "生活就像一盒巧克力。",
}


@pytest.mark.parametrize(
    ("src_lang", "tgt_lang"),
    [
        ("pt", "es"),
        ("es", "en"),
        ("en", "zh"),
        ("zh", "pt"),
    ],
)
def test_translate_sentences(src_lang, tgt_lang):
    sentence = TEXTS[src_lang]
    translated_sentences = translate_sentence(
        sentence, src_lang=src_lang, tgt_lang=tgt_lang
    )
    assert translated_sentences
    assert len(translated_sentences) == 1
    assert translated_sentences[0] == TEXTS[tgt_lang]
