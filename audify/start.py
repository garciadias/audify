from pathlib import Path

import click
from TTS.api import TTS

from audify.text_to_speech import EpubSynthesizer

MODULE_PATH = Path(__file__).resolve().parents[1]


@click.command()
# Path to the epub file to be synthesized is required as default parameter
@click.argument(
    "epub_path",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="tts_models/multilingual/multi-dataset/xtts_v2",
    help="Path to the TTS model.",
)
@click.option(
    "--language",
    "-l",
    type=click.Choice(["en", "es", "pt"], case_sensitive=False),
    default="en",
    help="Language of the synthesized audiobook.",
)
@click.option(
    "--voice",
    "-v",
    type=str,
    default="data/Jennifer_16khz.wav",
    help="Path to the speaker's voice.",
)
@click.option(
    "--list-languages",
    "-ll",
    is_flag=True,
)
@click.option(
    "--list-models",
    "-lm",
    is_flag=True,
)
def main(
    file_path: str,
    language: str,
    voice: str,
    list_languages: bool,
    list_models: bool,
    model: str,
):
    book_synthesizer = EpubSynthesizer(
        file_path, language=language, speaker=voice, model_name=model
    )
    if list_languages:
        print("====================")
        print("Available languages:")
        print("====================")
        print(", ".join(book_synthesizer.model.languages))
    elif list_models:
        print("=================")
        print("Available models:")
        print("=================")
        print(", ".join(TTS().list_models().list_models()))
    else:
        book_synthesizer.synthesize()


if __name__ == "__main__":
    main()
