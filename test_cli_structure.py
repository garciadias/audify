#!/usr/bin/env python3
"""Verify CLI structure."""

from audify.cli import cli

# Check if the commands are registered
print("CLI group commands:")
if hasattr(cli, "commands"):
    print(f"  {cli.commands.keys()}")
else:
    print("  No commands attribute")

# Check the __dict__
print("\nCLI object attributes:")
for attr in dir(cli):
    if not attr.startswith("_"):
        print(f"  {attr}")

# Try to get help text
from click.testing import CliRunner

runner = CliRunner()

print("\n\n=== HELP OUTPUT ===")
result = runner.invoke(cli, ["--help"])
print(result.output)

print("\n\n=== LIST-TASKS OUTPUT ===")
result = runner.invoke(cli, ["list-tasks"])
print(f"Exit code: {result.exit_code}")
print(f"Output:\n{result.output}")
if result.exception:
    print(f"Exception: {result.exception}")
    import traceback

    traceback.print_exception(
        type(result.exception), result.exception, result.exception.__traceback__
    )

print("\n\n=== VALIDATE-PROMPT OUTPUT ===")
import tempfile
from pathlib import Path

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write("A valid prompt with sufficient content for testing purposes")
    f.flush()
    result = runner.invoke(cli, ["validate-prompt", f.name])
print(f"Exit code: {result.exit_code}")
print(f"Output:\n{result.output}")
if result.exception:
    print(f"Exception: {result.exception}")
    import traceback

    traceback.print_exception(
        type(result.exception), result.exception, result.exception.__traceback__
    )
Path(f.name).unlink()
