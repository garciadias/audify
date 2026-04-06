# CLI Reference

## Global

```bash
audify --version       # Show version
audify --help          # Show help
```

## `audify run`

Basic TTS conversion without LLM processing.

```text
audify run [OPTIONS] FILE_PATH
```

### Arguments

| Argument    | Description            | Default |
|-------------|------------------------|---------|
| `FILE_PATH` | Path to EPUB or PDF    | `./`    |

### Options

| Option                    | Short | Description                        | Default   |
|---------------------------|-------|------------------------------------|-----------|
| `--language`              | `-l`  | Audio language                     | `en`      |
| `--model`                 | `-m`  | TTS model                          | `kokoro`  |
| `--voice`                 | `-v`  | Voice name                         | `af_bella`|
| `--translate`             | `-t`  | Translate to language              |           |
| `--save-text`             | `-st` | Save extracted text                |           |
| `--output`                | `-o`  | Output directory                   |           |
| `--tts-provider`          | `-tp` | TTS provider                       | `kokoro`  |
| `--list-languages`        | `-ll` | List available languages           |           |
| `--list-models`           | `-lm` | List available TTS models          |           |
| `--list-voices`           | `-lv` | List available voices              |           |
| `--list-tts-providers`    | `-ltp`| List TTS providers                 |           |
| `--create-voice-samples`  | `-cvs`| Create sample M4B with all voices  |           |
| `--max-samples`           | `-ms` | Max voice samples                  | `5`       |
| `-y`                      | `-y`  | Skip confirmation                  |           |

## `audify audiobook`

LLM-powered audiobook generation.

```text
audify audiobook [OPTIONS] PATH
```

### Arguments

| Argument | Description                        |
|----------|------------------------------------|
| `PATH`   | Path to EPUB, PDF, or directory    |

### Options

| Option           | Short | Description                          | Default          |
|------------------|-------|--------------------------------------|------------------|
| `--language`     | `-l`  | Audio language                       | `en`             |
| `--voice`        | `-v`  | Voice name                           | `af_bella`       |
| `--voice-model`  | `-vm` | TTS model                            | `kokoro`         |
| `--translate`    | `-t`  | Translate to language                |                  |
| `--save-scripts` | `-st` | Save generated scripts               |                  |
| `--llm-base-url` |       | LLM API base URL                     | `localhost:11434`|
| `--llm-model`    | `-m`  | LLM model (`api:` prefix for cloud)  | `magistral:24b`  |
| `--max-chapters` | `-mc` | Max chapters (EPUB only)             |                  |
| `--confirm`      | `-y`  | Ask for confirmation                 |                  |
| `--output`       | `-o`  | Output directory                     |                  |
| `--tts-provider` | `-tp` | TTS provider                         | `kokoro`         |
| `--task`         | `-T`  | Task name (audiobook, podcast, etc.) |                  |
| `--prompt-file`  | `-pf` | Custom prompt file path              |                  |

## `audify convert`

Unified command combining `run` and `audiobook` functionality.

```text
audify convert [OPTIONS] INPUT_PATH
```

Supports all options from both `run` and `audiobook` commands, plus `--task` and `--prompt-file`.

## `audify list-tasks`

List all registered transformation tasks.

```bash
audify list-tasks
```

## `audify validate-prompt`

Validate a custom prompt file.

```bash
audify validate-prompt my-prompt.txt
```
