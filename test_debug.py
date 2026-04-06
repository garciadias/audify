#!/usr/bin/env python3
"""Debug test for CLI commands."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from audify.cli import cli


def test_list_tasks():
    """Test list-tasks command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["list-tasks"])
    print(f"Exit code: {result.exit_code}")
    print(f"Output:\n{result.output}")
    if result.exception:
        print(f"Exception: {result.exception}")
        import traceback

        traceback.print_exception(
            type(result.exception), result.exception, result.exception.__traceback__
        )


def test_validate_prompt():
    """Test validate-prompt command."""
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("A valid prompt with sufficient content for testing purposes")
        f.flush()
        result = runner.invoke(cli, ["validate-prompt", f.name])
    print(f"\nExit code: {result.exit_code}")
    print(f"Output:\n{result.output}")
    if result.exception:
        print(f"Exception: {result.exception}")
        import traceback

        traceback.print_exception(
            type(result.exception), result.exception, result.exception.__traceback__
        )
    Path(f.name).unlink()


if __name__ == "__main__":
    test_list_tasks()
    test_validate_prompt()
