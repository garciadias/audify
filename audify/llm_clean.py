import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Initialize the T5 model and tokenizer for text cleaning
def initialize_llm_model(model_name="t5-base"):
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
    # Initial regex cleaning as in utils.py
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    cleaned = cleaned.replace("@", "a")
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r" ([.,!?;:¿¡-])", r"\1", cleaned)
    cleaned = re.sub(r"[\s\[\]{}()<>/\\#]", " ", cleaned)
    cleaned = re.sub(r" +", " ", cleaned)

    # Prepare prompt for LLM
    prompt = f"Clean and refine the following text: {cleaned}\n\n"

    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Decode output
    outputs = model.generate(inputs, max_length=512, num_beams=4)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return decoded


# Example usage:
if __name__ == "__main__":
    # Assume 'text_to_clean' is the raw text extracted from PDF
    cleaned_text = clean_with_llm("Your r@w text goe s here .")
    print(cleaned_text)
