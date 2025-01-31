from pathlib import Path

from audify.text_to_speech import EpubSynthesizer

MODULE_PATH = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    book_synthesizer = EpubSynthesizer(f"{MODULE_PATH}/data/federated.epub")
    book_synthesizer.synthesize()
