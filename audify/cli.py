#!/usr/bin/env python3
"""
Audify CLI - Convert ebooks and PDFs to audiobooks using AI text-to-speech.

Unified CLI entry point supporting:
- Direct TTS conversion (--task direct)
- LLM-powered audiobook generation (--task audiobook, podcast, etc.)
- Directory processing
- Multiple TTS providers
"""

import importlib.metadata
import logging
import os
import warnings
from pathlib import Path

import click
import requests

import audify.convert as convert_module
from audify.audiobook_creator import DirectoryAudiobookCreator
from audify.prompts.manager import PromptManager
from audify.prompts.tasks import TaskRegistry
from audify.text_to_speech import VoiceSamplesSynthesizer
from audify.utils.api_config import get_tts_config
from audify.utils.constants import (
    AVAILABLE_LANGUAGES,
    AVAILABLE_TTS_PROVIDERS,
    DEFAULT_LANGUAGE_LIST,
    DEFAULT_TTS_PROVIDER,
    KOKORO_API_BASE_URL,
    OLLAMA_API_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
)
from audify.utils.logging_utils import configure_cli_logging
from audify.utils.text import get_file_extension

# Ignore UserWarning from pkg_resources about package metadata
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

try:
    __version__ = importlib.metadata.version("audify")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"


@click.group(
    invoke_without_command=True,
    context_settings={"allow_extra_args": True},
)
@click.option(
    "--language",
    "-l",
    type=click.Choice(DEFAULT_LANGUAGE_LIST, case_sensitive=False),
    default="en",
    help="Language of the synthesized audio.",
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
    "--save-text",
    "-st",
    is_flag=True,
    help="Save the extracted text to a file.",
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
    "-lm",
    type=str,
    default=None,
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
    help="Skip confirmation before proceeding.",
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
    default="audiobook",
    help="Task name for prompt selection (e.g., 'audiobook', 'podcast', "
    "'summary', 'meditation', 'lecture', 'direct'). Defaults to 'audiobook'.",
)
@click.option(
    "--prompt-file",
    "-pf",
    type=click.Path(exists=True),
    default=None,
    help="Path to a custom prompt file. Overrides --task prompt.",
)
@click.option(
    "--list-languages",
    "-ll",
    is_flag=True,
    help="List available languages.",
)
@click.option(
    "--list-models",
    "-lmm",
    is_flag=True,
    help="List available TTS models.",
)
@click.option(
    "--list-voices",
    "-lv",
    is_flag=True,
    help="List available TTS voices.",
)
@click.option(
    "--list-tts-providers",
    "-ltp",
    is_flag=True,
    help="List available TTS providers and their configuration requirements.",
)
@click.option(
    "--create-voice-samples",
    "-cvs",
    is_flag=True,
    help="Create a sample M4B audiobook with all available model-voice combinations.",
)
@click.option(
    "--max-samples",
    "-ms",
    type=int,
    default=5,
    help="Maximum number of voice samples to create when using --create-voice-samples.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed log messages in terminal.",
)
@click.argument("path", nargs=-1, type=click.Path())
@click.version_option(__version__, "--version", "-V", message="audify %(version)s")
@click.pass_context
def cli(
    ctx: click.Context,
    language: str,
    voice_model: str,
    translate: str | None,
    voice: str,
    save_text: bool,
    llm_base_url: str,
    llm_model: str,
    max_chapters: int | None,
    confirm: bool,
    output: str | None,
    tts_provider: str,
    task: str,
    prompt_file: str | None,
    list_languages: bool,
    list_models: bool,
    list_voices: bool,
    list_tts_providers: bool,
    create_voice_samples: bool,
    max_samples: int,
    verbose: bool,
    path: tuple[str, ...],
):
    """Audify: Convert ebooks and PDFs to audiobooks using AI text-to-speech."""
    # If a subcommand is invoked, let it handle the logic
    if ctx.invoked_subcommand is not None:
        return

    path_str = path[0] if path else None

    # Configure logging based on verbose flag
    configure_cli_logging(verbose=verbose)
    logger = logging.getLogger(__name__)

    # Handle early-exit listing flags
    try:
        terminal_width = os.get_terminal_size()[0]
    except OSError:
        terminal_width = 80

    if list_tts_providers:
        click.echo("=" * terminal_width)
        click.echo("Available TTS Providers".center(terminal_width))
        click.echo("=" * terminal_width)
        click.echo("\nProvider\tStatus\t\tConfiguration")
        click.echo("-" * terminal_width)

        provider_info = {
            "kokoro": {
                "name": "Kokoro (Local)",
                "config": "KOKORO_API_URL (default: http://localhost:8887/v1)",
            },
            "openai": {
                "name": "OpenAI TTS",
                "config": "OPENAI_API_KEY, OPENAI_TTS_MODEL, OPENAI_TTS_VOICE",
            },
            "aws": {
                "name": "AWS Polly",
                "config": "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
                "AWS_REGION, AWS_POLLY_VOICE",
            },
            "google": {
                "name": "Google Cloud TTS",
                "config": "GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_TTS_VOICE",
            },
        }

        for provider in AVAILABLE_TTS_PROVIDERS:
            try:
                config = get_tts_config(provider=provider, language=language)
                status = "Available" if config.is_available() else "Not configured"
            except Exception:
                status = "Not available"

            info = provider_info.get(provider, {"name": provider, "config": "N/A"})
            click.echo(f"{info['name']:<16}\t{status:<16}\t{info['config']}")

        click.echo("\n" + "=" * terminal_width)
        click.echo(
            "Set TTS_PROVIDER environment variable or use --tts-provider/-tp flag"
        )
        click.echo("=" * terminal_width)
        return

    if create_voice_samples:
        click.echo("=" * terminal_width)
        click.echo("Creating Voice Samples M4B".center(terminal_width))
        click.echo("=" * terminal_width)
        synthesizer = VoiceSamplesSynthesizer(
            language=language,
            translate=translate,
            max_samples=max_samples,
            output_dir=output,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )
        synthesizer.synthesize()
        return

    if list_languages:
        click.echo("=" * terminal_width)
        click.echo("Available languages:".center(terminal_width))
        click.echo("=" * terminal_width)
        click.echo("Language\tCode")
        click.echo("--------\t----")
        for lang, code in AVAILABLE_LANGUAGES.items():
            click.echo(f"{lang:<10}\t{code}")
        click.echo("=" * terminal_width)
        return

    if list_models:
        click.echo("=" * terminal_width)
        click.echo("Available models:".center(terminal_width))
        click.echo("=" * terminal_width)
        try:
            response = requests.get(f"{KOKORO_API_BASE_URL}/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            model_names = sorted(model.get("id") for model in models if "id" in model)
            click.echo("\n".join(model_names))
        except Exception as e:
            click.echo(f"Error fetching models from Kokoro API: {e}")
        return

    if list_voices:
        click.echo("=" * terminal_width)
        click.echo(
            f"Available voices for {tts_provider.upper()}:".center(terminal_width)
        )
        click.echo("=" * terminal_width)
        try:
            config = get_tts_config(provider=tts_provider, language=language)
            voices = config.get_available_voices()
            if voices:
                if tts_provider == "kokoro":
                    # Group Kokoro voices by prefix for better organization
                    voice_groups: dict[str, list[str]] = {}
                    for v in voices:
                        prefix = v.split("_")[0] if "_" in v else "other"
                        if prefix not in voice_groups:
                            voice_groups[prefix] = []
                        voice_groups[prefix].append(v)

                    for prefix in sorted(voice_groups.keys()):
                        click.echo(f"\n{prefix.upper()} voices:")
                        for v in sorted(voice_groups[prefix]):
                            click.echo(f"  {v}")
                else:
                    # For other providers, just list voices
                    click.echo(f"\nVoices for {tts_provider}:")
                    for v in sorted(voices):
                        click.echo(f"  {v}")
            else:
                click.echo(f"No voices found for {tts_provider}.")
        except Exception as e:
            click.echo(f"Error fetching voices from {tts_provider}: {e}")
        return

    # If no path provided and no listing flag, show help
    if path_str is None:
        click.echo(ctx.get_help())
        return

    effective_llm_model = llm_model or OLLAMA_DEFAULT_MODEL

    # Normal conversion path
    path_obj = Path(path_str)

    # Validate path exists
    if not path_obj.exists():
        click.echo(f"Error: Path '{path_str}' does not exist.", err=True)
        ctx.exit(1)

    logger.info("=" * terminal_width)

    # Check if path is a directory
    if path_obj.is_dir():
        logger.info("Directory Mode: Processing multiple files".center(terminal_width))
        logger.info("=" * terminal_width)
        logger.info(f"Source directory: {path_str}")
        logger.info(f"Language: {language}")
        logger.info(f"LLM Model: {effective_llm_model}")
        logger.info(f"TTS Provider: {tts_provider}")
        logger.info(f"Task: {task}")
        if prompt_file:
            logger.info(f"Prompt file: {prompt_file}")
        if translate:
            logger.info(f"Translation: {language} -> {translate}")

        logger.info("=" * terminal_width)

        try:
            # Create directory audiobook creator
            dir_creator = DirectoryAudiobookCreator(
                directory_path=path_str,
                language=language,
                voice=voice,
                model_name=voice_model,
                translate=translate,
                save_text=save_text,
                llm_base_url=llm_base_url,
                llm_model=effective_llm_model,
                confirm=not confirm,
                output_dir=output,
                tts_provider=tts_provider,
                task=task,
                prompt_file=prompt_file,
            )
            # Generate the audiobook
            output_path = dir_creator.synthesize()

            logger.info("\n" + "=" * terminal_width)
            logger.info("Directory audiobook creation complete!")
            logger.info(f"Output directory: {output_path}")
            logger.info("=" * terminal_width)

        except KeyboardInterrupt:
            logger.info("\n\nDirectory audiobook creation cancelled by user.")
            return
        except Exception as e:
            logger.error(f"\nError: {e}")
            logger.error("Please check your configuration and try again.")
            if "Could not connect to LLM" in str(e):
                logger.error("\nTip: Make sure Ollama is running:")
                logger.error("  ollama serve")
                logger.error(f"  ollama pull {effective_llm_model}")
            return

    else:
        # Single file mode
        file_extension = get_file_extension(path_str)

        # Show configuration
        logger.info(f"Source file: {path_str}")
        logger.info(f"Language: {language}")
        logger.info(f"LLM Model: {effective_llm_model}")
        logger.info(f"TTS Provider: {tts_provider}")
        logger.info(f"Task: {task}")
        if prompt_file:
            logger.info(f"Prompt file: {prompt_file}")
        if translate:
            logger.info(f"Translation: {language} -> {translate}")
        if max_chapters:
            logger.info(f"Max episodes: {max_chapters}")

        logger.info("=" * terminal_width)

        try:
            creator = convert_module.get_creator(
                file_extension=file_extension,
                path=path_str,
                language=language,
                voice=voice,
                model_name=voice_model,
                translate=translate,
                save_text=save_text,
                llm_base_url=llm_base_url,
                llm_model=effective_llm_model,
                max_chapters=max_chapters,
                confirm=not confirm,
                output_dir=output,
                tts_provider=tts_provider,
                task=task,
                prompt_file=prompt_file,
            )
            # Generate the audiobook
            output_path = creator.synthesize()

            logger.info("\n" + "=" * terminal_width)
            logger.info("Audiobook creation complete!")
            logger.info(f"Output directory: {output_path}")
            logger.info("=" * terminal_width)

        except KeyboardInterrupt:
            logger.info("\n\nAudiobook creation cancelled by user.")
        except Exception as e:
            logger.error(f"\nError: {e}")
            logger.error("Please check your configuration and try again.")
            if "Could not connect to LLM" in str(e):
                logger.error("\nTip: Make sure Ollama is running:")
                logger.error("  ollama serve")
                logger.error(f"  ollama pull {effective_llm_model}")


@cli.command("list-tasks")
def list_tasks():
    """List all available transformation tasks."""
    try:
        terminal_width = os.get_terminal_size()[0]
    except OSError:
        terminal_width = 80

    tasks = TaskRegistry.get_all()

    click.echo("=" * terminal_width)
    click.echo("Available Tasks".center(terminal_width))
    click.echo("=" * terminal_width)
    click.echo()
    click.echo(f"{'Task':<15} {'Requires LLM':<15} {'Output':<12} Description")
    click.echo("-" * terminal_width)

    descriptions = {
        "direct": "Direct TTS conversion, no LLM processing",
        "audiobook": "Transform text into an engaging audiobook script",
        "podcast": "Transform text into a comprehensive talk/podcast",
        "summary": "Create a concise audio summary",
        "meditation": "Transform text into a guided meditation script",
        "lecture": "Transform text into an engaging classroom lecture",
    }

    for name in sorted(tasks.keys()):
        task = tasks[name]
        llm = "Yes" if task.requires_llm else "No"
        desc = descriptions.get(name, "Custom task")
        click.echo(f"  {name:<13} {llm:<15} {task.output_structure:<12} {desc}")

    click.echo()
    click.echo("=" * terminal_width)
    click.echo("Usage: audify <input> --task <task-name>")
    click.echo("       audify <input> --prompt-file <path>")
    click.echo("=" * terminal_width)


@cli.command("validate-prompt")
@click.argument("prompt_file", type=click.Path(exists=True))
def validate_prompt(prompt_file: str):
    """Validate a custom prompt file."""
    manager = PromptManager()
    prompt = manager.load_prompt_file(prompt_file)
    is_valid, message = manager.validate_prompt(prompt)

    if is_valid:
        click.echo(f"Prompt file is valid: {prompt_file}")
        click.echo(f"  Length: {len(prompt)} characters")
        click.echo(f"  Preview: {prompt[:100]}...")
    else:
        raise click.ClickException(f"Prompt validation failed: {message}")


if __name__ == "__main__":
    cli()
