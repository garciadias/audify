"""Prompt manager for loading and managing prompts."""

from pathlib import Path
from typing import Optional

from audify.utils.logging_utils import setup_logging

logger = setup_logging(module_name=__name__)

BUILTIN_PROMPTS_DIR = Path(__file__).parent / "builtin"


class PromptManager:
    """Manages loading and resolving prompts from various sources."""

    def get_builtin_prompt(self, task_name: str) -> str:
        """Load a built-in prompt by task name.

        Args:
            task_name: Name of the built-in task (e.g., "audiobook", "podcast").

        Returns:
            The prompt text.

        Raises:
            FileNotFoundError: If the built-in prompt file doesn't exist.
        """
        prompt_path = BUILTIN_PROMPTS_DIR / f"{task_name}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Built-in prompt not found: {task_name}. "
                f"Available prompts: {self.list_builtin_prompts()}"
            )
        return prompt_path.read_text(encoding="utf-8")

    def load_prompt_file(self, path: str | Path) -> str:
        """Load a custom prompt from a file path.

        Args:
            path: Path to the prompt file.

        Returns:
            The prompt text.

        Raises:
            FileNotFoundError: If the prompt file doesn't exist.
            ValueError: If the prompt file is empty.
        """
        prompt_path = Path(path).expanduser().resolve()
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        content = prompt_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Prompt file is empty: {prompt_path}")
        return content

    def get_prompt(
        self,
        task: str,
        prompt_file: Optional[str | Path] = None,
    ) -> str:
        """Resolve the prompt for a given task.

        Priority: prompt_file > task's built-in prompt.

        Args:
            task: Task name (e.g., "audiobook", "podcast", "direct").
            prompt_file: Optional path to a custom prompt file.

        Returns:
            The resolved prompt text. Empty string for "direct" task.
        """
        if prompt_file:
            logger.info(f"Loading custom prompt from: {prompt_file}")
            return self.load_prompt_file(prompt_file)

        if task == "direct":
            return ""

        # Try to get from task registry first (avoids re-reading files)
        from audify.prompts.tasks import TaskRegistry

        task_config = TaskRegistry.get(task)
        if task_config:
            return task_config.prompt

        # Fallback: try loading as a builtin prompt file
        try:
            return self.get_builtin_prompt(task)
        except FileNotFoundError:
            raise ValueError(
                f"Unknown task '{task}' and no prompt file provided. "
                f"Available tasks: {TaskRegistry.list_tasks()}"
            )

    def list_builtin_prompts(self) -> list[str]:
        """List available built-in prompt names."""
        return sorted(
            p.stem for p in BUILTIN_PROMPTS_DIR.glob("*.txt") if p.stem != "__init__"
        )

    def validate_prompt(self, prompt: str) -> tuple[bool, str]:
        """Validate a prompt string.

        Returns:
            Tuple of (is_valid, message).
        """
        if not prompt or not prompt.strip():
            return False, "Prompt is empty."
        if len(prompt) < 10:
            return False, "Prompt is too short (minimum 10 characters)."
        return True, "Prompt is valid."
