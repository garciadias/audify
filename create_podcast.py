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

from audify.constants import DEFAULT_LANGUAGE_LIST
from audify.podcast_creator import PodcastEpubCreator, PodcastPdfCreator
from audify.utils import get_file_extension

MODULE_PATH = Path(__file__).resolve().parents[1]


@click.command()
@click.argument(
    "file_path",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--language",
    "-l",
    type=click.Choice(DEFAULT_LANGUAGE_LIST, case_sensitive=False),
    default="en",
    help="Language of the source text.",
)
@click.option(
    "--llm-model",
    "-lm",
    type=str,
    default="qwen3:30b",
    help="Local LLM model to use for podcast script generation.",
)
@click.option(
    "--llm-url",
    "-lu",
    type=str,
    default="http://localhost:11434",
    help="Base URL for the local LLM API (e.g., Ollama).",
)
@click.option(
    "--tts-model",
    "-tm",
    type=str,
    default=None,
    help="TTS model to use for speech synthesis.",
)
@click.option(
    "--translate",
    "-t",
    type=click.Choice(DEFAULT_LANGUAGE_LIST, case_sensitive=False),
    help="Translate the text to the specified language before TTS.",
    default=None,
)
@click.option(
    "--voice",
    "-v",
    type=str,
    default="af_bella",
    help="Path to the speaker's voice file.",
)
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["kokoro", "tts_models"], case_sensitive=False),
    default="kokoro",
    help="The TTS engine to use.",
)
@click.option(
    "--max-chapters",
    "-mc",
    type=int,
    help="Maximum number of chapters/episodes to create.",
    default=None,
)
@click.option(
    "--save-scripts/--no-save-scripts",
    default=True,
    help="Save generated podcast scripts to files.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompts.",
)
def main(
    file_path: str,
    language: str,
    llm_model: str,
    llm_url: str,
    tts_model: str | None,
    translate: str | None,
    voice: str,
    engine: str,
    max_chapters: int | None,
    save_scripts: bool,
    yes: bool,
):
    """Create podcast episodes from ebooks or PDFs using LLM and TTS."""

    terminal_width = os.get_terminal_size()[0]
    file_extension = get_file_extension(file_path)

    print("=" * terminal_width)

    if file_extension == ".epub":
        print("EPUB to Podcast Series".center(terminal_width))
        print("=" * terminal_width)

        creator_class = PodcastEpubCreator
        format_name = "EPUB"

    elif file_extension == ".pdf":
        print("PDF to Podcast Episode".center(terminal_width))
        print("=" * terminal_width)

        creator_class = PodcastPdfCreator
        format_name = "PDF"

    else:
        print(f"Error: Unsupported file format '{file_extension}'")
        print("Supported formats: .epub, .pdf")
        return

    # Show configuration
    print(f"Source file: {file_path}")
    print(f"Format: {format_name}")
    print(f"Language: {language}")
    print(f"LLM Model: {llm_model}")
    print(f"LLM URL: {llm_url}")
    print(f"TTS Engine: {engine}")
    if tts_model:
        print(f"TTS Model: {tts_model}")
    if translate:
        print(f"Translation: {language} -> {translate}")
    if max_chapters:
        print(f"Max episodes: {max_chapters}")

    print("=" * terminal_width)

    try:
        # Create the podcast creator
        creator = creator_class(
            path=file_path,
            language=language,
            speaker=voice,
            model_name=tts_model,
            translate=translate,
            save_text=save_scripts,
            engine=engine,
            llm_base_url=llm_url,
            llm_model=llm_model,
            max_chapters=max_chapters,
            confirm=not yes,
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
