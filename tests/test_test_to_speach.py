from pathlib import Path

from audify import text_to_speech

MODULE_PATH = Path(__file__).parent


def test_text_to_speech():
    text_to_speech.sentence_to_speech(sentence="Hello, World!")
