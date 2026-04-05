# Usage

## Basic TTS Conversion

Convert ebooks directly to audio without LLM processing:

```bash
# EPUB to audiobook
audify run book.epub

# PDF to audio
audify run document.pdf

# Specify language
audify run book.epub --language pt

# Translate content (English to Spanish)
audify run book.epub --language en --translate es

# Choose TTS provider
audify run book.epub --tts-provider openai
```

## LLM-Powered Audiobook Generation

Use an LLM to transform text into engaging audiobook scripts before TTS:

```bash
# Default audiobook style
audify audiobook book.epub

# Limit chapters
audify audiobook book.epub --max-chapters 5

# Custom voice and language
audify audiobook book.epub --voice af_bella --language en

# With translation
audify audiobook book.epub --translate pt
```

### Using commercial LLM APIs

```bash
# DeepSeek (cost-effective)
audify audiobook book.epub -m "api:deepseek/deepseek-chat"

# Claude (high quality)
audify audiobook book.epub -m "api:anthropic/claude-3-5-sonnet-20240620"

# GPT-4
audify audiobook book.epub -m "api:openai/gpt-4-turbo-preview"

# Gemini
audify audiobook book.epub -m "api:gemini/gemini-1.5-pro"
```

See [Commercial APIs](commercial-apis.md) for API key setup.

## Task System

Use the `--task` flag to control how the LLM transforms your text:

```bash
# Audiobook style (default)
audify audiobook book.epub --task audiobook

# Podcast/lecture style
audify audiobook book.epub --task podcast

# Summary
audify audiobook book.epub --task summary

# Guided meditation
audify audiobook book.epub --task meditation

# Classroom lecture
audify audiobook book.epub --task lecture

# Custom prompt file
audify audiobook book.epub --prompt-file my-prompt.txt
```

See [Tasks](tasks.md) for details on creating custom tasks.

## Directory Processing

Process multiple files into a single audiobook:

```bash
# All supported files in a directory
audify audiobook path/to/articles/

# With translation
audify audiobook path/to/articles/ --translate es
```

**Supported file types**: EPUB, PDF, TXT, MD

Each file becomes a separate episode with a synthesized title, and all episodes are combined into a single M4B audiobook with chapter markers.

## Unified Convert Command

The `convert` command unifies `run` and `audiobook` functionality:

```bash
# Direct TTS (equivalent to audify run)
audify convert book.epub --task direct

# Audiobook (equivalent to audify audiobook)
audify convert book.epub --task audiobook

# Podcast style
audify convert book.epub --task podcast
```

## Listing Available Options

```bash
# List available tasks
audify list-tasks

# Validate a custom prompt file
audify validate-prompt my-prompt.txt

# List languages
audify run --list-languages

# List TTS voices
audify run --list-voices

# List TTS models
audify run --list-models

# List TTS providers
audify run --list-tts-providers
```

## Output Structure

### Basic TTS (`audify run`)

```text
data/output/[book_name]/
    chapters.txt           # Metadata
    cover.jpg              # Cover image (EPUB)
    chapters_001.mp3       # Chapter audio files
    chapters_002.mp3
    book_name.m4b          # Final audiobook
```

### LLM Audiobook (`audify audiobook`)

```text
audiobooks/[book_name]/
    episodes/
        episode_001.mp3
        episode_002.mp3
    scripts/
        episode_001_script.txt
        original_text_001.txt
    chapters.txt
    book_name.m4b
```
