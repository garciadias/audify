import os
import warnings
from pathlib import Path

import click
import requests

from audify.text_to_speech import EpubSynthesizer, PdfSynthesizer
from audify.utils.constants import (
    AVAILABLE_LANGUAGES,
    DEFAULT_LANGUAGE_LIST,
    KOKORO_API_BASE_URL,
)
from audify.utils.text import get_file_extension

# Ignore UserWarning from pkg_resources about package metadata
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

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
    default="af_bella",
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
        print("=" * terminal_width)
        print("Available languages:".center(terminal_width))
        print("=" * terminal_width)
        print("Language\tCode")
        print("--------\t----")
        for lang, code in AVAILABLE_LANGUAGES.items():
            print(f"{lang:<10}\t{code}")
        print("=" * terminal_width)
    elif list_models:
        print("=" * terminal_width)
        print("Available models:".center(terminal_width))
        print("=" * terminal_width)
        try:
            response = requests.get(f"{KOKORO_API_BASE_URL}/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            model_names = sorted(model.get("id") for model in models if "id" in model)
            print("\n".join(model_names))
        except requests.RequestException as e:
            print(f"Error fetching models from Kokoro API: {e}")
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
                confirm=not y,
            )  # type: ignore
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
            )  # type: ignore
            synthesizer.synthesize()
        else:
            raise ValueError("Unsupported file format")


if __name__ == "__main__":
    main()
