import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from audify.utils import break_text_into_sentences


# Initialize the T5 model and tokenizer for text cleaning
def initialize_llm_model(model_name="google/flan-t5-small"):
    """
    Initializes the T5 model for text-to-text tasks.

    Args:
        model_name: Name of the pre-trained T5 model to use.

    Returns:
        Model and tokenizer loaded onto the appropriate device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# Load T5 model and tokenizer
model, tokenizer = initialize_llm_model()


def clean_with_llm(raw_text: str) -> str:
    """
    Enhances text cleaning using both regex substitutions and LLM-based refinement.

    Args:
        raw_text: Raw text extracted from PDF.

    Returns:
        Cleaned text with improved quality.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = (
        "Identify URLs and replace them with 'URL'. Remove long string chains "
        "and tables. Clean text by eliminating spaces before punctuation, removing "
        "unnecessary special characters, and converting it to plain text format.\n\n"
    )
    inputs = tokenizer.encode(f"Input: {raw_text}\n\nOutput:", return_tensors="pt").to(
        device
    )
    outputs = model.generate(inputs, max_length=512, num_beams=4)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return decoded.replace(prompt, "")


# Example usage:
if __name__ == "__main__":
    # Assume 'text_to_clean' is the raw text extracted from PDF
    from audify.pdf_read import PdfReader
    from audify.utils import clean_text

    pdf_path = "data/test2.pdf"
    original_text = PdfReader(path=pdf_path).get_cleaned_text()
    text_splits = break_text_into_sentences(
        original_text, min_length=200, max_length=380
    )
    with open("./data/original.txt", "w+") as f:
        f.write("\n".join(text_splits))
    cleaned_text = []
    for sentence in text_splits:
        cleaned_text.append(clean_with_llm(sentence))
    final_text = "\n".join(cleaned_text)
    final_text = clean_text(final_text)
    with open("./data/clean.txt", "w+") as f:
        f.write(final_text)
