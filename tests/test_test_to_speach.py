from pathlib import Path
from unittest.mock import patch

from audify.text_to_speech import EpubSynthesizer

MODULE_PATH = Path(__file__).parent


@patch("audify.text_to_speech.TTS")
def test_epub_synthesizer_init(mock_tts, tmp_path):
    dummy_epub_path = tmp_path / "dummy.epub"
    dummy_epub_path.touch()
    synthesizer = EpubSynthesizer(dummy_epub_path, "en")
    assert synthesizer.language == "en"
    assert synthesizer.reader is not None
