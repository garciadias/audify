# CLI Reference

## Synopsis

```bash
audify [OPTIONS] [PATH]
audify list-tasks
audify validate-prompt PROMPT_FILE
```

## `audify [OPTIONS] [PATH]`

Main conversion command. When `PATH` is omitted, prints help. When `PATH` is a
directory, processes all supported files inside it.

### Arguments

| Argument | Description                                |
|----------|--------------------------------------------|
| `PATH`   | Path to an EPUB, PDF, or directory         |

### Conversion options

| Option           | Short  | Description                                                         | Default             |
|------------------|--------|---------------------------------------------------------------------|---------------------|
| `--task`         | `-T`   | Task name: `direct`, `audiobook`, `podcast`, `summary`, `meditation`, `lecture` | `audiobook` |
| `--prompt-file`  | `-pf`  | Path to a custom prompt file (overrides `--task`)                   |                     |
| `--tts-provider` | `-tp`  | TTS provider: `kokoro`, `openai`, `aws`, `google`           | `kokoro`            |
| `--voice`        | `-v`   | Voice name or ID for the selected TTS provider                      | `af_bella`          |
| `--voice-model`  | `-vm`  | TTS model name or path                                              | `kokoro`            |
| `--language`     | `-l`   | Language code for synthesized audio                                 | `en`                |
| `--translate`    | `-t`   | Translate text to this language before synthesis                    |                     |
| `--llm-model`    | `-m`   | LLM model name. Prefix with `api:` for cloud (e.g. `api:openai/gpt-4o`) | `magistral:24b` |
| `--llm-base-url` |        | Base URL for local Ollama API                                       | `http://localhost:11434` |
| `--max-chapters` | `-mc`  | Maximum chapters to process (EPUB only)                             |                     |
| `--output`       | `-o`   | Output directory                                                    |                     |
| `--save-text`    | `-st`  | Save extracted or generated text to a file                          |                     |
| `--confirm`      | `-y`   | Skip confirmation prompts                                           |                     |
| `--verbose`      |        | Show detailed log output                                            |                     |

### Info / listing options

| Option                   | Short  | Description                                               |
|--------------------------|--------|-----------------------------------------------------------|
| `--list-languages`       | `-ll`  | Print all supported language codes                        |
| `--list-models`          | `-lmm` | List TTS models available from Kokoro API                 |
| `--list-voices`          | `-lv`  | List voices available from the selected TTS provider      |
| `--list-tts-providers`   | `-ltp` | Show all TTS providers and their configuration status     |
| `--create-voice-samples` | `-cvs` | Generate an M4B sample file with all available voices     |
| `--max-samples`          | `-ms`  | Maximum number of voice samples to create                 | `5` |
| `--version`              | `-V`   | Print version and exit                                    |
| `--help`                 |        | Show help and exit                                        |

### Examples

```bash
# Direct TTS (no LLM)
audify book.epub --task direct

# LLM-enhanced audiobook (default task)
audify book.epub

# Podcast-style audio using OpenAI TTS
audify book.epub --task podcast --tts-provider openai

# Use a cloud LLM with local Kokoro TTS
audify book.epub --llm-model api:openai/gpt-4o

# Use a custom prompt
audify book.epub --prompt-file my-prompt.txt

# Process all EPUBs in a directory
audify ./my-books/ --task audiobook --tts-provider kokoro

# List all voices for Kokoro
audify --list-voices --tts-provider kokoro

# Translate to Spanish
audify book.epub --translate es
```

## `audify list-tasks`

Print all registered transformation tasks with their description and whether they
require an LLM.

```bash
audify list-tasks
```

## `audify validate-prompt`

Validate a custom prompt file before use.

```bash
audify validate-prompt my-prompt.txt
```

### Arguments

| Argument      | Description                       |
|---------------|-----------------------------------|
| `PROMPT_FILE` | Path to the prompt file to check  |
