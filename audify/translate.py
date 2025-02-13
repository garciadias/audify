from transformers import M2M100ForConditionalGeneration as M2MCG
from transformers import M2M100Tokenizer as M2MT


def translate_sentences(
    sentences: list[str],
    model: M2MCG = M2MCG.from_pretrained("facebook/m2m100_418M"),
    tokenizer: M2MT = M2MT.from_pretrained("facebook/m2m100_418M"),
    src_lang: str = "zh",
    tgt_lang: str = "en",
) -> list[str]:
    tokenizer.src_lang = src_lang
    encoded_sentences = tokenizer(sentences, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_sentences, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
