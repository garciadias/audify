#!/usr/bin/env python3
"""
Audify CLI - Convert ebooks and PDFs to audiobooks using AI text-to-speech.

Provides two main commands:
- run: Basic TTS conversion of EPUB/PDF files
- audiobook: LLM-powered audiobook generation
"""

import importlib.metadata

import click

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
cli.add_command(run_command, name="run")
cli.add_command(audiobook_command, name="audiobook")


if __name__ == "__main__":
    cli()
