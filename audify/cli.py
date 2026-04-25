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
import shutil
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


class SubcommandAwareGroup(click.Group):
    """Custom Group that handles subcommand names consumed as the PATH argument.

    With invoke_without_command=True and an optional PATH argument, Click
    parses e.g. "list-tasks" as the PATH value instead of recognising it
    as a subcommand.  This override detects the situation and re-routes.
    """

    def invoke(self, ctx: click.Context) -> None:
        path_value = ctx.params.get("path")
        if path_value and path_value in self.commands:
            cmd = self.commands[path_value]
            cmd_ctx = cmd.make_context(
                path_value, list(ctx.protected_args + ctx.args), parent=ctx
            )
            ctx.params["path"] = None
            ctx.invoked_subcommand = path_value
            with cmd_ctx:
                cmd.invoke(cmd_ctx)
            return
        super().invoke(ctx)


# Suppress verbose output from external libraries
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Ignore UserWarning from pkg_resources about package metadata
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

try:
    __version__ = importlib.metadata.version("audify-cli")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"


_CONTAINER_DATA_ROOT = Path("/app/data")
_CONTAINER_INPUT_ROOT = _CONTAINER_DATA_ROOT / "input"
_CONTAINER_OUTPUT_ROOT = _CONTAINER_DATA_ROOT / "output"


def _is_container_runtime() -> bool:
    """Return True when running in a container with shared /app/data mount."""
    return Path("/.dockerenv").exists() and _CONTAINER_DATA_ROOT.exists()


def _resolve_input_path_for_runtime(path_str: str, logger: logging.Logger) -> str:
    """Resolve host-like paths to container-mounted paths when possible."""
    raw_path = Path(path_str).expanduser()
    if raw_path.exists() or not _is_container_runtime():
        return str(raw_path)

    raw_string = str(raw_path)
    candidates: list[Path] = []

    # Common host path pattern: /.../data/<relative>
    marker = "/data/"
    if marker in raw_string:
        suffix = raw_string.split(marker, 1)[1]
        candidates.append(_CONTAINER_DATA_ROOT / suffix)

    if raw_path.is_absolute():
        candidates.append(_CONTAINER_DATA_ROOT / "ebooks" / raw_path.name)
        candidates.append(Path("/app") / raw_path.name)
    else:
        candidates.append(Path("/app") / raw_path)
        candidates.append(_CONTAINER_DATA_ROOT / raw_path)
        candidates.append(_CONTAINER_DATA_ROOT / "ebooks" / raw_path)

    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            logger.info(
                f"Resolved input path '{path_str}' to '{candidate_str}' "
                "for container runtime"
            )
            return candidate_str

    return str(raw_path)


def _stage_input_to_host_data(path_obj: Path, logger: logging.Logger) -> Path:
    """Copy non-shared container inputs into /app/data/input for host visibility."""
    if not _is_container_runtime() or not path_obj.exists():
        return path_obj

    resolved = path_obj.resolve()
    if resolved.is_relative_to(_CONTAINER_DATA_ROOT):
        return resolved

    try:
        _CONTAINER_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        staged_path = _CONTAINER_INPUT_ROOT / resolved.name

        if resolved.is_dir():
            shutil.copytree(resolved, staged_path, dirs_exist_ok=True)
        else:
            shutil.copy2(resolved, staged_path)
    except OSError as e:
        logger.warning(
            f"Failed to stage input to {_CONTAINER_INPUT_ROOT}: {e}. "
            "Falling back to original path."
        )
        return path_obj

    logger.info(f"Staged input for host visibility: {resolved} -> {staged_path}")
    return staged_path


def _resolve_output_path_for_runtime(
    output: str | None,
    logger: logging.Logger,
) -> str | None:
    """Map output paths to /app/data when running in a container."""
    if output is None:
        if _is_container_runtime():
            _CONTAINER_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            return str(_CONTAINER_OUTPUT_ROOT)
        return None

    output_path = Path(output).expanduser()
    if output_path.exists() or not _is_container_runtime():
        return str(output_path)

    output_string = str(output_path)
    marker = "/data/"
    if marker in output_string:
        suffix = output_string.split(marker, 1)[1]
        mapped = _CONTAINER_DATA_ROOT / suffix
        logger.info(f"Resolved output path '{output}' to '{mapped}'")
        return str(mapped)

    if not output_path.is_absolute():
        mapped = _CONTAINER_OUTPUT_ROOT / output_path
        logger.info(f"Resolved relative output path '{output}' to '{mapped}'")
        return str(mapped)

    return str(output_path)


def _ensure_output_synced_to_host_data(
    output_path: Path,
    logger: logging.Logger,
) -> Path:
    """Ensure output artifacts are available under /app/data/output."""
    if not _is_container_runtime() or not output_path.exists():
        return output_path

    resolved = output_path.resolve()
    if resolved.is_relative_to(_CONTAINER_DATA_ROOT):
        return resolved

    try:
        _CONTAINER_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        target = _CONTAINER_OUTPUT_ROOT / resolved.name

        if resolved.is_dir():
            shutil.copytree(resolved, target, dirs_exist_ok=True)
        else:
            shutil.copy2(resolved, target)
    except OSError as e:
        logger.warning(
            f"Failed to sync output to {_CONTAINER_OUTPUT_ROOT}: {e}. "
            "Returning original path."
        )
        return output_path

    logger.info(f"Copied output artifact to host-visible path: {target}")
    return target


def _contains_audio_artifacts(output_path: Path) -> bool:
    """Return True when output path contains generated audio artifacts."""
    if not output_path.exists():
        return False

    if output_path.is_file():
        return output_path.suffix.lower() in {".mp3", ".wav", ".m4b"}

    for pattern in ("*.mp3", "*.wav", "*.m4b"):
        if any(output_path.rglob(pattern)):
            return True

    return False


@click.group(
    cls=SubcommandAwareGroup,
    invoke_without_command=True,
    context_settings={
        "allow_extra_args": True,
        "allow_interspersed_args": True,
    },
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
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
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
    "--process-only",
    is_flag=True,
    default=False,
    help="Only extract text and generate scripts (no TTS synthesis). "
    "Use --synthesize-only later to produce audio.",
)
@click.option(
    "--synthesize-only",
    is_flag=True,
    default=False,
    help="Skip text extraction and script generation; synthesise "
    "audio from previously saved scripts (requires a prior "
    "--process-only run).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed log messages in terminal.",
)
@click.argument("path", required=False, type=click.Path())
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
    process_only: bool,
    synthesize_only: bool,
    verbose: bool,
    path: str | None,
):
    """Audify: Convert ebooks and PDFs to audiobooks using AI text-to-speech."""
    if ctx.invoked_subcommand is not None:
        return

    path_str = path

    configure_cli_logging(verbose=verbose)
    logger = logging.getLogger(__name__)

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
            from audify.utils.api_config import _retry_request

            def _fetch_models():
                resp = requests.get(f"{KOKORO_API_BASE_URL}/models", timeout=5)
                resp.raise_for_status()
                return resp.json().get("data", [])

            models = _retry_request(
                _fetch_models,
                api_name=f"Kokoro API ({KOKORO_API_BASE_URL}/models)",
            )
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
                    click.echo(f"\nVoices for {tts_provider}:")
                    for v in sorted(voices):
                        click.echo(f"  {v}")
            else:
                click.echo(f"No voices found for {tts_provider}.")
        except Exception as e:
            click.echo(f"Error fetching voices from {tts_provider}: {e}")
        return

    if path_str is None:
        click.echo(ctx.get_help())
        return

    path_str = _resolve_input_path_for_runtime(path_str, logger)
    output = _resolve_output_path_for_runtime(output, logger)

    effective_llm_model = llm_model or OLLAMA_DEFAULT_MODEL

    # Resolve processing mode from CLI flags
    if process_only and synthesize_only:
        click.echo(
            "Error: --process-only and --synthesize-only are mutually exclusive.",
            err=True,
        )
        ctx.exit(1)
    if process_only:
        mode = "process"
    elif synthesize_only:
        mode = "synthesize"
    else:
        mode = "full"

    path_obj = Path(path_str)

    if not path_obj.exists():
        click.echo(f"Error: Path '{path_str}' does not exist.", err=True)
        ctx.exit(1)

    path_obj = _stage_input_to_host_data(path_obj, logger)
    path_str = str(path_obj)

    click.echo("=" * terminal_width)

    if path_obj.is_dir():
        click.echo("Directory Mode: Processing multiple files".center(terminal_width))
        click.echo("=" * terminal_width)
        click.echo(f"Source directory: {path_str}")
        click.echo(f"Language: {language}")
        click.echo(f"LLM Model: {effective_llm_model}")
        click.echo(f"TTS Provider: {tts_provider}")
        click.echo(f"Task: {task}")
        click.echo(f"Mode: {mode}")
        if prompt_file:
            click.echo(f"Prompt file: {prompt_file}")
        if translate:
            click.echo(f"Translation: {language} -> {translate}")

        click.echo("=" * terminal_width)
        logger.info("Directory Mode: Processing multiple files")
        logger.info(f"Source directory: {path_str}")
        logger.info(f"Language: {language}")
        logger.info(f"LLM Model: {effective_llm_model}")
        logger.info(f"TTS Provider: {tts_provider}")
        logger.info(f"Task: {task}")
        logger.info(f"Mode: {mode}")
        if prompt_file:
            logger.info(f"Prompt file: {prompt_file}")
        if translate:
            logger.info(f"Translation: {language} -> {translate}")

        try:
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
                mode=mode,
            )
            output_path = dir_creator.synthesize()
            output_path = _ensure_output_synced_to_host_data(
                Path(output_path),
                logger,
            )

            if output_path.exists() and not _contains_audio_artifacts(output_path):
                message = (
                    "No audio artifacts were generated in the output directory. "
                    "Check TTS/LLM logs for errors and verify provider settings."
                )
                logger.error(message)
                click.echo(f"Error: {message}", err=True)
                click.echo(f"Output directory: {output_path}", err=True)
                raise SystemExit(1)

            # Find the M4B file in the output directory
            m4b_files = list(output_path.glob("*.m4b"))
            m4b_path = m4b_files[0] if m4b_files else None

            click.echo("\n" + "=" * terminal_width)
            click.echo("✓ Audiobook creation complete!".center(terminal_width))
            click.echo("=" * terminal_width)
            if m4b_path:
                click.echo(f"M4B file: {m4b_path}")
            click.echo(f"Output directory: {output_path}")
            click.echo("=" * terminal_width)

            logger.info("\n" + "=" * terminal_width)
            logger.info("Directory audiobook creation complete!")
            logger.info(f"Output directory: {output_path}")
            if m4b_path:
                logger.info(f"M4B file: {m4b_path}")
            logger.info("=" * terminal_width)

        except KeyboardInterrupt:
            logger.info("Directory audiobook creation cancelled by user.")
            raise SystemExit(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error("Please check your configuration and try again.")
            if "Could not connect to LLM" in str(e):
                logger.error("Tip: Make sure Ollama is running: ollama serve")
                logger.error(f"  ollama pull {effective_llm_model}")
            raise SystemExit(1)

    else:
        file_extension = get_file_extension(path_str)

        click.echo(f"Source file: {path_str}")
        click.echo(f"Language: {language}")
        click.echo(f"LLM Model: {effective_llm_model}")
        click.echo(f"TTS Provider: {tts_provider}")
        click.echo(f"Task: {task}")
        click.echo(f"Mode: {mode}")
        if prompt_file:
            click.echo(f"Prompt file: {prompt_file}")
        if translate:
            click.echo(f"Translation: {language} -> {translate}")
        if max_chapters:
            click.echo(f"Max episodes: {max_chapters}")

        click.echo("=" * terminal_width)

        logger.info(f"Source file: {path_str}")
        logger.info(f"Language: {language}")
        logger.info(f"LLM Model: {effective_llm_model}")
        logger.info(f"TTS Provider: {tts_provider}")
        logger.info(f"Task: {task}")
        logger.info(f"Mode: {mode}")
        if prompt_file:
            logger.info(f"Prompt file: {prompt_file}")
        if translate:
            logger.info(f"Translation: {language} -> {translate}")
        if max_chapters:
            logger.info(f"Max episodes: {max_chapters}")

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
                mode=mode,
            )
            output_path = creator.synthesize()
            output_path = _ensure_output_synced_to_host_data(
                Path(output_path),
                logger,
            )

            if output_path.exists() and not _contains_audio_artifacts(output_path):
                message = (
                    "No audio artifacts were generated in the output directory. "
                    "Check TTS/LLM logs for errors and verify provider settings."
                )
                logger.error(message)
                click.echo(f"Error: {message}", err=True)
                click.echo(f"Output directory: {output_path}", err=True)
                raise SystemExit(1)

            # Find the M4B file in the output directory
            m4b_files = list(output_path.glob("*.m4b"))
            m4b_path = m4b_files[0] if m4b_files else None

            click.echo("\n" + "=" * terminal_width)
            click.echo("✓ Audiobook creation complete!".center(terminal_width))
            click.echo("=" * terminal_width)
            if m4b_path:
                click.echo(f"M4B file: {m4b_path}")
            click.echo(f"Output directory: {output_path}")
            click.echo("=" * terminal_width)

            logger.info("\n" + "=" * terminal_width)
            logger.info("Audiobook creation complete!")
            logger.info(f"Output directory: {output_path}")
            if m4b_path:
                logger.info(f"M4B file: {m4b_path}")
            logger.info("=" * terminal_width)

        except KeyboardInterrupt:
            logger.info("Audiobook creation cancelled by user.")
            raise SystemExit(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error("Please check your configuration and try again.")
            if "Could not connect to LLM" in str(e):
                logger.error("Tip: Make sure Ollama is running: ollama serve")
                logger.error(f"  ollama pull {effective_llm_model}")
            raise SystemExit(1)


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
