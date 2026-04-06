# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Flexible prompt system with task registry (`--task`, `--prompt-file`)
- Built-in tasks: audiobook, podcast, summary, meditation, lecture
- `audify list-tasks` command to list available tasks
- `audify validate-prompt` command to validate custom prompt files
- `audify convert` unified command
- ReadTheDocs documentation
- CI/CD pipeline for PyPI releases

### Changed

- `LLMClient` now has a generic `generate_script()` method accepting custom prompts
- `AudiobookCreator` accepts `task` and `prompt_file` parameters

## [0.1.0] - 2024-01-01

### Added

- Initial release
- EPUB and PDF to audiobook conversion
- Multiple TTS providers: Kokoro, OpenAI, AWS Polly, Google Cloud TTS, Qwen-TTS
- LLM-powered audiobook script generation (Ollama + commercial APIs)
- REST API for programmatic access
- Directory processing for multi-file audiobooks
- Translation support
- M4B audiobook output with chapter markers
