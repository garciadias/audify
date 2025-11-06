#!/usr/bin/env python3
"""
Podcast Creator CLI - Generate podcasts from ebooks and PDFs using LLM and TTS.

This script creates podcast episodes from ebook chapters or PDF content by:
1. Extracting text from the source file
2. Using a local LLM to generate podcast scripts
3. Converting scripts to speech using TTS
"""

import os
from pathlib import Path

import click

from audify.audiobook_creator import (
    PodcastCreator,
    PodcastEpubCreator,
    PodcastPdfCreator,
)
from audify.utils.constants import (
    DEFAULT_LANGUAGE_LIST,
    OLLAMA_API_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
)
from audify.utils.text import get_file_extension

MODULE_PATH = Path(__file__).resolve().parents[1]


def get_creator(
    file_extension: str,
    path: str,
    language: str,
    voice: str,
    model_name: str,
    translate: str | None,
    save_text: bool,
    llm_base_url: str,
    llm_model: str,
    max_chapters: int | None,
    confirm: bool,
) -> PodcastCreator:
    """Get the appropriate PodcastCreator subclass based on file extension.

    Args:
        file_extension: The file extension (e.g., '.epub', '.pdf').

    Returns:
        The corresponding PodcastCreator subclass.

    Raises:
        TypeError: If the file extension is unsupported.
    """
    if file_extension == ".epub":
        return PodcastEpubCreator(
            path=path,
            language=language,
            voice=voice,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            max_chapters=max_chapters,
            confirm=confirm,
        )
    elif file_extension == ".pdf":
        # remove max_chapters for PDF
        return PodcastPdfCreator(
            path=path,
            language=language,
            voice=voice,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            confirm=confirm,
        )
    else:
        raise TypeError(f"Unsupported file format '{file_extension}'")


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--language",
    "-l",
    type=click.Choice(DEFAULT_LANGUAGE_LIST, case_sensitive=False),
    default="en",
    help="Language of the synthesized podcast.",
)
@click.option(
    "--model-name",
    "-m",
    type=str,
    default="kokoro",
    help="Path to the TTS model or 'kokoro' to use Kokoro TTS API.",
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
    help="Path to the speaker's voice or voice name if using Kokoro API.",
)
@click.option(
    "--save-scripts",
    "-st",
    is_flag=True,
    help="Save the text extraction to a file.",
)
@click.option(
    "--llm-base-url",
    type=str,
    default=OLLAMA_API_BASE_URL,
    help=f"Base URL for the LLM API (default: {OLLAMA_API_BASE_URL}).",
)
@click.option(
    "--llm-model",
    type=str,
    default=OLLAMA_DEFAULT_MODEL,
    help=f"The LLM model to use (default: {OLLAMA_DEFAULT_MODEL}).",
)
@click.option(
    "--max-chapters",
    "-mc",
    type=int,
    help="Maximum number of chapters/episodes to create (only for EPUB).",
    default=None,
)
@click.option(
    "--confirm",
    "-y",
    is_flag=True,
    default=False,
    help="Ask for confirmation before proceeding.",
)
def main(
    path: str,
    language: str,
    voice: str,
    model_name: str,
    translate: str | None,
    save_scripts: bool,
    llm_base_url: str,
    llm_model: str,
    max_chapters: int | None,
    confirm: bool,
):
    """Create podcast episodes from ebooks or PDFs using LLM and TTS."""

    terminal_width = os.get_terminal_size()[0]
    file_extension = get_file_extension(path)

    print("=" * terminal_width)

    # Show configuration
    print(f"Source file: {path}")
    print(f"Language: {language}")
    print(f"LLM Model: {llm_model}")
    if translate:
        print(f"Translation: {language} -> {translate}")
    if max_chapters:
        print(f"Max episodes: {max_chapters}")

    print("=" * terminal_width)

    try:
        creator = get_creator(
            file_extension=file_extension,
            path=path,
            language=language,
            voice=voice,
            model_name=model_name,
            translate=translate,
            save_text=save_scripts,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            max_chapters=max_chapters,
            confirm=not confirm,
        )
        # Generate the podcast
        output_path = creator.synthesize()

        print("\n" + "=" * terminal_width)
        print("Podcast creation complete!")
        print(f"Output directory: {output_path}")
        print("=" * terminal_width)

    except KeyboardInterrupt:
        print("\n\nPodcast creation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your configuration and try again.")
        if "Could not connect to LLM" in str(e):
            print("\nTip: Make sure Ollama is running:")
            print("  ollama serve")
            print(f"  ollama pull {llm_model}")


if __name__ == "__main__":
    main()
