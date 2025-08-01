import os
from pathlib import Path

import click

from audify.constants import DEFAULT_LANGUAGE_LIST
from audify.domain.interface import Synthesizer
from audify.text_to_speech import EpubSynthesizer, InspectSynthesizer, PdfSynthesizer
from audify.utils import get_file_extension

MODULE_PATH = Path(__file__).resolve().parents[1]


@click.command()
@click.argument(
    "file_path",
    type=click.Path(exists=True),
    default="./",
)
@click.option(
    "--language",
    "-l",
    type=click.Choice(DEFAULT_LANGUAGE_LIST, case_sensitive=False),
    default="en",
    help="Language of the synthesized audiobook.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="kokoro",
    help="Path to the TTS model.",
)
@click.option(
    "--translate",
    "-t",
    type=click.Choice(DEFAULT_LANGUAGE_LIST, case_sensitive=False),
    help="Translate the text to the specified language.",
    default=None,
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
    help="List available languages.",
)
@click.option(
    "--list-models",
    "-lm",
    is_flag=True,
    help="List available TTS models.",
)
@click.option(
    "--save-text",
    "-st",
    is_flag=True,
    help="Save the text extraction to a file.",
)
@click.option(
    "--engine",
    "-e",
    type=str,
    default="kokoro",
    help="The TTS engine to use (tts_models or kokoro).",
)
@click.option(
    "--y",
    "-y",
    is_flag=True,
    help="Skip confirmation for Epub synthesis.",
)
def main(
    file_path: str,
    language: str,
    model: str,
    translate: str | None,
    voice: str,
    list_languages: bool,
    list_models: bool,
    save_text: bool,
    engine: str = "kokoro",
    y: bool = False,
):
    terminal_width = os.get_terminal_size()[0]
    if list_languages:
        synthesizer: Synthesizer = InspectSynthesizer()
        print("=" * terminal_width)
        print("Available languages:".center(terminal_width))
        print("=" * terminal_width)
        print(", ".join(synthesizer.model.languages))
    elif list_models:
        synthesizer = InspectSynthesizer()
        print("=" * terminal_width)
        print("Available models:".center(terminal_width))
        print("=" * terminal_width)
        print("\n".join(synthesizer.model.models))
    else:
        if get_file_extension(file_path) == ".epub":
            print("=" * terminal_width)
            print("Epub to Audiobook".center(terminal_width))
            print("=" * terminal_width)
            synthesizer = EpubSynthesizer(
                file_path,
                language=language,
                speaker=voice,
                model_name=model,
                translate=translate,
                save_text=save_text,
                engine=engine,
                confirm=not y,
            )
            synthesizer.synthesize()
        elif get_file_extension(file_path) == ".pdf":
            print("==========")
            print("PDF to mp3")
            print("==========")
            synthesizer = PdfSynthesizer(
                file_path,
                language=language,
                speaker=voice,
                model_name=model,
                translate=translate,
                save_text=save_text,
                engine=engine,
            )
            synthesizer.synthesize()
        else:
            raise ValueError("Unsupported file format")


if __name__ == "__main__":
    main()
