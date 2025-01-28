# %%
from pathlib import Path

from audify.synthesizer import BookSynthesizer

MODULE_PATH = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    # %%
    book_path = f"{MODULE_PATH}/data/test.epub"
    synthesizer = BookSynthesizer(book_path)
    synthesizer.synthesize()
