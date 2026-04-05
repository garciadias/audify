import logging
import os

import click
import requests

from audify.text_to_speech import (
    EpubSynthesizer,
    PdfSynthesizer,
    VoiceSamplesSynthesizer,
)
from audify.utils.api_config import get_tts_config
from audify.utils.constants import (
    AVAILABLE_LANGUAGES,
    AVAILABLE_TTS_PROVIDERS,
    DEFAULT_LANGUAGE_LIST,
    DEFAULT_TTS_PROVIDER,
    KOKORO_API_BASE_URL,
    OLLAMA_API_BASE_URL,
)

# Configure logging
from audify.utils.logging_utils import configure_cli_logging
from audify.utils.text import get_file_extension

logger = logging.getLogger("audify")

# Stream handler will be added by configure_cli_logging if verbose flag is set


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
    "--list-voices",
    "-lv",
    is_flag=True,
    help="List available TTS voices.",
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
    "--list-tts-providers",
    "-ltp",
    is_flag=True,
    help="List available TTS providers and their configuration requirements.",
)
@click.option(
    "--llm-model",
    type=str,
    default=None,
    help="LLM model to use for translation. For Ollama: model name. "
    "For commercial APIs: 'api:model_name' (e.g., 'api:deepseek/deepseek-chat').",
)
@click.option(
    "--llm-base-url",
    type=str,
    default=OLLAMA_API_BASE_URL,
    help=f"Base URL for the LLM API (default: {OLLAMA_API_BASE_URL}).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output to terminal (also logs to file).",
)
def main(
    file_path: str,
    language: str,
    model: str,
    translate: str | None,
    voice: str,
    list_languages: bool,
    list_models: bool,
    list_voices: bool,
    save_text: bool,
    create_voice_samples: bool,
    max_samples: int | None = None,
    y: bool = False,
    output: str | None = None,
    tts_provider: str = DEFAULT_TTS_PROVIDER,
    list_tts_providers: bool = False,
    llm_model: str | None = None,
    llm_base_url: str = OLLAMA_API_BASE_URL,
    verbose: bool = False,
):
    """Basic TTS conversion of EPUB/PDF files to audio (no LLM)."""
    # Configure logging based on verbose flag
    configure_cli_logging(verbose=verbose)

    terminal_width = os.get_terminal_size()[0]
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
    elif create_voice_samples:
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
    elif list_languages:
        click.echo("=" * terminal_width)
        click.echo("Available languages:".center(terminal_width))
        click.echo("=" * terminal_width)
        click.echo("Language\tCode")
        click.echo("-------\t----")
        for lang, code in AVAILABLE_LANGUAGES.items():
            click.echo(f"{lang:<10}\t{code}")
        click.echo("=" * terminal_width)
    elif list_models:
        click.echo("=" * terminal_width)
        click.echo("Available models:".center(terminal_width))
        click.echo("=" * terminal_width)
        try:
            response = requests.get(f"{KOKORO_API_BASE_URL}/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            model_names = sorted(model.get("id") for model in models if "id" in model)
            click.echo("\n".join(model_names))
        except requests.RequestException as e:
            click.echo(f"Error fetching models from Kokoro API: {e}")
    elif list_voices:
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
    else:
        if get_file_extension(file_path) == ".epub":
            logger.info("=" * terminal_width)
            logger.info("Epub to Audiobook".center(terminal_width))
            logger.info("=" * terminal_width)
            logger.info(f"TTS Provider: {tts_provider}")
            synthesizer = EpubSynthesizer(
                file_path,
                language=language,
                speaker=voice,
                model_name=model,
                translate=translate,
                save_text=save_text,
                confirm=not y,
                output_dir=output,
                tts_provider=tts_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
            )  # type: ignore
            synthesizer.synthesize()
        elif get_file_extension(file_path) == ".pdf":
            logger.info("==========")
            logger.info("PDF to mp3")
            logger.info("==========")
            logger.info(f"TTS Provider: {tts_provider}")
            synthesizer = PdfSynthesizer(
                file_path,
                language=language,
                speaker=voice,
                model_name=model,
                translate=translate,
                save_text=save_text,
                output_dir=output,
                tts_provider=tts_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
            )  # type: ignore
            synthesizer.synthesize()
        else:
            raise ValueError("Unsupported file format")


if __name__ == "__main__":
    main()
