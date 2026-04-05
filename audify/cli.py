#!/usr/bin/env python3
"""
Audify CLI - Convert ebooks and PDFs to audiobooks using AI text-to-speech.

Provides commands:
- run: Basic TTS conversion of EPUB/PDF files (no LLM)
- audiobook: LLM-powered audiobook generation
- list-tasks: List available transformation tasks
- validate-prompt: Validate a custom prompt file
"""

import importlib.metadata
import os
import sys

import click

from audify.convert import convert as convert_command
from audify.create_audiobook import main as audiobook_command

# Import existing CLI commands
from audify.start import main as run_command

try:
    __version__ = importlib.metadata.version("audify")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"


@click.group()
@click.version_option(__version__, "--version", "-V", message="audify %(version)s")
def cli():
    """Audify: Convert ebooks and PDFs to audiobooks using AI text-to-speech."""
    pass


# Add subcommands
cli.add_command(convert_command, name="convert")
cli.add_command(run_command, name="run")
cli.add_command(audiobook_command, name="audiobook")


@cli.command("list-tasks")
def list_tasks():
    """List all available transformation tasks."""
    from audify.prompts.tasks import TaskRegistry

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
    click.echo("Usage: audify audiobook <input> --task <task-name>")
    click.echo("       audify audiobook <input> --prompt-file <path>")
    click.echo("=" * terminal_width)


@cli.command("validate-prompt")
@click.argument("prompt_file", type=click.Path(exists=True))
def validate_prompt(prompt_file: str):
    """Validate a custom prompt file."""
    from audify.prompts.manager import PromptManager

    manager = PromptManager()

    try:
        prompt = manager.load_prompt_file(prompt_file)
        is_valid, message = manager.validate_prompt(prompt)

        if is_valid:
            click.echo(f"Prompt file is valid: {prompt_file}")
            click.echo(f"  Length: {len(prompt)} characters")
            click.echo(f"  Preview: {prompt[:100]}...")
        else:
            click.echo(f"Prompt file validation failed: {message}")
            sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
