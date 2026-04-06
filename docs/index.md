# Audify Documentation

Convert ebooks and PDFs to audiobooks using AI text-to-speech.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

usage
tasks
configuration
rest-api
```

```{toctree}
:maxdepth: 2
:caption: Reference

cli-reference
architecture
commercial-apis
```

```{toctree}
:maxdepth: 2
:caption: Development

contributing
changelog
```

## What is Audify?

Audify is a pipeline and CLI tool that transforms written content (EPUB, PDF, TXT, MD) into high-quality audiobooks using:

- **Multiple TTS Providers** -- Kokoro (local), Qwen-TTS (local), OpenAI, AWS Polly, Google Cloud TTS
- **LLM-powered audio generation** -- Transform text into audiobook scripts, podcasts, lectures, summaries, and more
- **Flexible task system** -- Use built-in prompts or provide your own custom prompts

## Quick Example

```bash
# Install
pip install audify

# Convert an EPUB to audiobook (direct TTS)
audify run book.epub

# Generate an LLM-enhanced audiobook
audify audiobook book.epub

# Use a different task (podcast-style)
audify audiobook book.epub --task podcast

# Use a custom prompt
audify audiobook book.epub --prompt-file my-prompt.txt
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
