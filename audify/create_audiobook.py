#!/usr/bin/env python3
"""
Audiobook Creator CLI - Generate audiobooks from ebooks and PDFs using LLM and TTS.

This script creates audiobook episodes from ebook chapters or PDF content by:
1. Extracting text from the source file
2. Using a local LLM to generate audiobook scripts
3. Converting scripts to speech using TTS
"""

import os
from pathlib import Path

import click

from audify.audiobook_creator import (
    AudiobookCreator,
    AudiobookEpubCreator,
    AudiobookPdfCreator,
    DirectoryAudiobookCreator,
)
from audify.utils.constants import (
    AVAILABLE_TTS_PROVIDERS,
    DEFAULT_LANGUAGE_LIST,
    DEFAULT_TTS_PROVIDER,
    OLLAMA_API_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
)
from audify.utils.text import get_file_extension


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
    output_dir: str | None = None,
    tts_provider: str | None = None,
    task: str | None = None,
    prompt_file: str | None = None,
) -> AudiobookCreator:
    """Get the appropriate AudiobookCreator subclass based on file extension.

    Args:
        file_extension: The file extension (e.g., '.epub', '.pdf').
        task: Task name for prompt selection (e.g., 'audiobook', 'podcast').
        prompt_file: Path to a custom prompt file.

    Returns:
        The corresponding AudiobookCreator subclass.

    Raises:
        TypeError: If the file extension is unsupported.
    """
    if file_extension == ".epub":
        return AudiobookEpubCreator(
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
            output_dir=output_dir,
            tts_provider=tts_provider,
            task=task,
            prompt_file=prompt_file,
        )
    elif file_extension == ".pdf":
        # remove max_chapters for PDF
        return AudiobookPdfCreator(
            path=path,
            language=language,
            voice=voice,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            confirm=confirm,
            output_dir=output_dir,
            tts_provider=tts_provider,
            task=task,
            prompt_file=prompt_file,
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
    help="Language of the synthesized audiobook.",
)
@click.option(
    "--voice-model",
    "-vm",
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
    "-m",
    type=str,
    default=OLLAMA_DEFAULT_MODEL,
    help=f"The LLM model to use. For Ollama: model name "
    f"(default: {OLLAMA_DEFAULT_MODEL}). For commercial APIs: "
    f"'api:model_name' (e.g., 'api:deepseek/deepseek-chat', "
    f"'api:anthropic/claude-3-sonnet-20240229', 'api:openai/gpt-4'). "
    f"Requires API keys in .keys file or environment variables.",
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
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory or file path for the result.",
)
@click.option(
    "--tts-provider",
    "-tp",
    type=click.Choice(AVAILABLE_TTS_PROVIDERS, case_sensitive=False),
    default=DEFAULT_TTS_PROVIDER,
    help=f"TTS provider to use (default: {DEFAULT_TTS_PROVIDER}). "
    "Options: kokoro (local), openai, aws (Polly), google (Cloud TTS).",
)
@click.option(
    "--task",
    "-T",
    type=str,
    default=None,
    help="Task name for prompt selection (e.g., 'audiobook', 'podcast', "
    "'summary', 'meditation', 'lecture'). Defaults to 'audiobook'.",
)
@click.option(
    "--prompt-file",
    "-pf",
    type=click.Path(exists=True),
    default=None,
    help="Path to a custom prompt file. Overrides --task prompt.",
)
def main(
    path: str,
    language: str,
    voice: str,
    voice_model: str,
    translate: str | None,
    save_scripts: bool,
    llm_base_url: str,
    llm_model: str,
    max_chapters: int | None,
    confirm: bool,
    output: str | None,
    tts_provider: str,
    task: str | None,
    prompt_file: str | None,
):
    """Create audiobook episodes from ebooks or PDFs using LLM and TTS."""

    try:
        terminal_width = os.get_terminal_size()[0]
    except OSError:
        terminal_width = 80  # Default width when no terminal is available
    path_obj = Path(path)

    print("=" * terminal_width)

    # Check if path is a directory
    if path_obj.is_dir():
        print("Directory Mode: Processing multiple files".center(terminal_width))
        print("=" * terminal_width)
        print(f"Source directory: {path}")
        print(f"Language: {language}")
        print(f"LLM Model: {llm_model}")
        print(f"TTS Provider: {tts_provider}")
        if task:
            print(f"Task: {task}")
        if prompt_file:
            print(f"Prompt file: {prompt_file}")
        if translate:
            print(f"Translation: {language} -> {translate}")

        print("=" * terminal_width)

        try:
            # Create directory audiobook creator
            dir_creator = DirectoryAudiobookCreator(
                directory_path=path,
                language=language,
                voice=voice,
                model_name=voice_model,
                translate=translate,
                save_text=save_scripts,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                confirm=not confirm,
                output_dir=output,
                tts_provider=tts_provider,
                task=task,
                prompt_file=prompt_file,
            )
            # Generate the audiobook
            output_path = dir_creator.synthesize()

            print("\n" + "=" * terminal_width)
            print("Directory audiobook creation complete!")
            print(f"Output directory: {output_path}")
            print("=" * terminal_width)

        except KeyboardInterrupt:
            print("\n\nDirectory audiobook creation cancelled by user.")
            return
        except Exception as e:
            print(f"\nError: {e}")
            print("Please check your configuration and try again.")
            if "Could not connect to LLM" in str(e):
                print("\nTip: Make sure Ollama is running:")
                print("  ollama serve")
                print(f"  ollama pull {llm_model}")
            return

    else:
        # Single file mode
        file_extension = get_file_extension(path)

        # Show configuration
        print(f"Source file: {path}")
        print(f"Language: {language}")
        print(f"LLM Model: {llm_model}")
        print(f"TTS Provider: {tts_provider}")
        if task:
            print(f"Task: {task}")
        if prompt_file:
            print(f"Prompt file: {prompt_file}")
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
                model_name=voice_model,
                translate=translate,
                save_text=save_scripts,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                max_chapters=max_chapters,
                confirm=not confirm,
                output_dir=output,
                tts_provider=tts_provider,
                task=task,
                prompt_file=prompt_file,
            )
            # Generate the audiobook
            output_path = creator.synthesize()

            print("\n" + "=" * terminal_width)
            print("Audiobook creation complete!")
            print(f"Output directory: {output_path}")
            print("=" * terminal_width)

        except KeyboardInterrupt:
            print("\n\nAudiobook creation cancelled by user.")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please check your configuration and try again.")
            if "Could not connect to LLM" in str(e):
                print("\nTip: Make sure Ollama is running:")
                print("  ollama serve")
                print(f"  ollama pull {llm_model}")


if __name__ == "__main__":
    main()
