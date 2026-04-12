"""Task configuration and registry for Audify prompt system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskConfig:
    """Configuration for a specific audio transformation task."""

    name: str
    prompt: str
    requires_llm: bool = True
    llm_params: dict = field(default_factory=dict)
    output_structure: str = "single"  # "single", "episodes", "chapters"

    def get_llm_params(self, **overrides) -> dict:
        """Get LLM parameters with optional overrides."""
        params = dict(self.llm_params)
        params.update(overrides)
        return params


class TaskRegistry:
    """Registry of available audio transformation tasks."""

    _tasks: dict[str, TaskConfig] = {}

    @classmethod
    def register(cls, config: TaskConfig) -> None:
        """Register a new task configuration.

        Args:
            config: The TaskConfig to register.
        """
        cls._tasks[config.name] = config

    @classmethod
    def get(cls, name: str) -> Optional[TaskConfig]:
        """Get a task configuration by name.

        Args:
            name: Task name to retrieve.

        Returns:
            The TaskConfig if found, None otherwise.
        """
        return cls._tasks.get(name)

    @classmethod
    def list_tasks(cls) -> list[str]:
        """List all registered task names.

        Returns:
            Sorted list of registered task names.
        """
        return sorted(cls._tasks.keys())

    @classmethod
    def get_all(cls) -> dict[str, TaskConfig]:
        """Get all registered task configurations.

        Returns:
            Dictionary of all registered tasks.
        """
        return dict(cls._tasks)

    @classmethod
    def _reset(cls) -> None:
        """Reset the registry (for testing).

        Clears all registered tasks from the registry.
        """
        cls._tasks = {}


def _register_builtin_tasks() -> None:
    """Register all built-in tasks."""
    from audify.prompts.manager import PromptManager

    manager = PromptManager()

    TaskRegistry.register(
        TaskConfig(
            name="direct",
            prompt="",
            requires_llm=False,
            output_structure="single",
        )
    )

    TaskRegistry.register(
        TaskConfig(
            name="audiobook",
            prompt=manager.get_builtin_prompt("audiobook"),
            requires_llm=True,
            llm_params={
                "temperature": 0.8,
                "top_p": 0.9,
                "repeat_penalty": 1.05,
                "seed": 428798,
                "top_k": 60,
                "num_predict": 4096,
                "num_ctx": 8 * 4096,
            },
            output_structure="single",
        )
    )

    TaskRegistry.register(
        TaskConfig(
            name="podcast",
            prompt=manager.get_builtin_prompt("podcast"),
            requires_llm=True,
            llm_params={
                "temperature": 0.9,
                "top_p": 0.95,
                "repeat_penalty": 1.05,
                "num_predict": 4096,
                "num_ctx": 8 * 4096,
            },
            output_structure="single",
        )
    )

    TaskRegistry.register(
        TaskConfig(
            name="summary",
            prompt=manager.get_builtin_prompt("summary"),
            requires_llm=True,
            llm_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 4096,
                "num_ctx": 8 * 4096,
            },
            output_structure="single",
        )
    )

    TaskRegistry.register(
        TaskConfig(
            name="meditation",
            prompt=manager.get_builtin_prompt("meditation"),
            requires_llm=True,
            llm_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 4096,
                "num_ctx": 8 * 4096,
            },
            output_structure="single",
        )
    )

    TaskRegistry.register(
        TaskConfig(
            name="lecture",
            prompt=manager.get_builtin_prompt("lecture"),
            requires_llm=True,
            llm_params={
                "temperature": 0.8,
                "top_p": 0.9,
                "num_predict": 4096,
                "num_ctx": 8 * 4096,
            },
            output_structure="single",
        )
    )


# Register built-in tasks on module import
_register_builtin_tasks()
