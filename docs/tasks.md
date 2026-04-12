# Task System

Audify's task system lets you control how the LLM transforms your text before TTS synthesis. Each task uses a different prompt that instructs the LLM to produce a specific style of output.

## Built-in Tasks

| Task         | Description                                     |
|--------------|-------------------------------------------------|
| `direct`     | No LLM processing, direct TTS conversion        |
| `audiobook`  | Transform into an engaging audiobook script      |
| `podcast`    | Transform into a comprehensive talk/podcast      |
| `summary`    | Create a concise audio summary                   |
| `meditation` | Transform into a guided meditation script        |
| `lecture`    | Transform into a classroom lecture               |

### List tasks

```bash
audify list-tasks
```

### Use a built-in task

```bash
audify audiobook book.epub --task podcast
audify audiobook book.epub --task summary
audify audiobook book.epub --task meditation
```

## Custom Prompt Files

Create a text file with LLM instructions and pass it with `--prompt-file`:

```bash
# Create a prompt file
cat > my-prompt.txt << 'EOF'
Transform the following content into a bedtime story.
Use gentle, soothing language. Simplify complex ideas.
Speak directly to the listener. End with a calming closing.
EOF

# Use the prompt
audify audiobook book.epub --prompt-file my-prompt.txt
```

The `--prompt-file` flag overrides any `--task` selection.

### Validate a prompt file

```bash
audify validate-prompt my-prompt.txt
```

## Writing Effective Prompts

A good custom prompt should:

1. **Define the role** -- Tell the LLM what kind of writer/narrator it is.
2. **Specify the output format** -- What should the final text look like?
3. **Set the tone** -- Professional, casual, calming, energetic, etc.
4. **Include constraints** -- What to include, what to exclude.
5. **Audio-specific rules** -- No citations, no URLs, no visual references, only speakable text.

### Example: Children's story prompt

```text
Transform the following content into an engaging story for children ages 8-12.
Use simple vocabulary and short sentences. Replace technical concepts with
relatable analogies. Include moments of wonder and excitement.

Rules:
- Use a warm, friendly narrator voice
- No citations, URLs, or academic references
- Replace numbers and statistics with descriptive language
- Only include text that should be spoken aloud
- Do not include any stage directions or meta-commentary
```

### Example: News briefing prompt

```text
Transform the following content into a concise news briefing.
Present the key facts clearly and objectively. Use the inverted
pyramid style: most important information first.

Structure:
- Lead: The most important fact in 1-2 sentences
- Body: Supporting details and context
- Background: Brief relevant history if needed

Rules:
- Write in present tense where possible
- No editorializing or opinion
- No citations or URLs
- Only include text that should be spoken aloud
```

## Programmatic Task Registration

Developers can register custom tasks in Python:

```python
from audify.prompts.tasks import TaskConfig, TaskRegistry

TaskRegistry.register(
    TaskConfig(
        name="bedtime_story",
        prompt="Transform text into a gentle bedtime story...",
        requires_llm=True,
        llm_params={"temperature": 0.7, "top_p": 0.9},
        output_structure="single",
    )
)
```
