#!/usr/bin/env python3
"""Direct test of the subcommands without pytest complexity."""

import sys
import tempfile
from pathlib import Path

print("Step 1: Importing CLI...")
try:
    from audify.cli import cli

    print("✓ CLI imported successfully")
except Exception as e:
    print(f"✗ Failed to import CLI: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\nStep 2: Checking if cli is a Click group...")
print(f"  cli type: {type(cli)}")
print(f"  cli.__class__.__name__: {cli.__class__.__name__}")

print("\nStep 3: Trying to get help (which lists commands)...")
from click.testing import CliRunner

runner = CliRunner()
help_result = runner.invoke(cli, ["--help"])
print(f"  Help exit code: {help_result.exit_code}")
if "list-tasks" in help_result.output:
    print("  ✓ list-tasks found in help output")
else:
    print("  ✗ list-tasks NOT found in help output")
if "validate-prompt" in help_result.output:
    print("  ✓ validate-prompt found in help output")
else:
    print("  ✗ validate-prompt NOT found in help output")

print("\n" + "=" * 60)
print("Testing list-tasks command")
print("=" * 60)

result = runner.invoke(cli, ["list-tasks"])

print(f"Exit code: {result.exit_code}")
print(f"Exit code == 0? {result.exit_code == 0}")
print(f"Output length: {len(result.output)}")
print(f"Full output:\n---\n{result.output}\n---")

if result.exception:
    print("\n⚠ EXCEPTION OCCURRED:")
    print(f"  Type: {type(result.exception).__name__}")
    print(f"  Message: {result.exception}")
    print("  Full traceback:")
    import traceback

    traceback.print_exception(
        type(result.exception), result.exception, result.exception.__traceback__
    )
else:
    print("\n✓ No exception")

checks = {
    "exit_code == 0": result.exit_code == 0,
    "'audiobook' in output": "audiobook" in result.output,
    "'podcast' in output": "podcast" in result.output,
    "'direct' in output": "direct" in result.output,
    "'summary' in output": "summary" in result.output,
    "'meditation' in output": "meditation" in result.output,
    "'lecture' in output": "lecture" in result.output,
}

print("\nChecks:")
for check, passed in checks.items():
    print(f"  {'✓' if passed else '✗'} {check}")

test1_pass = all(checks.values())
print(f"\n{'✓ TEST PASSED' if test1_pass else '✗ TEST FAILED'}")

print("\n" + "=" * 60)
print("Testing validate-prompt command")
print("=" * 60)

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    f.write("A valid prompt with sufficient content for testing purposes")
    f.flush()
    result = runner.invoke(cli, ["validate-prompt", f.name])
    temp_file = f.name

print(f"Exit code: {result.exit_code}")
print(f"Exit code == 0? {result.exit_code == 0}")
print(f"Full output:\n---\n{result.output}\n---")

if result.exception:
    print("\n⚠ EXCEPTION OCCURRED:")
    print(f"  Type: {type(result.exception).__name__}")
    print(f"  Message: {result.exception}")
    import traceback

    traceback.print_exception(
        type(result.exception), result.exception, result.exception.__traceback__
    )
else:
    print("\n✓ No exception")

checks2 = {
    "exit_code == 0": result.exit_code == 0,
    "'valid' in output.lower()": "valid" in result.output.lower(),
}

print("\nChecks:")
for check, passed in checks2.items():
    print(f"  {'✓' if passed else '✗'} {check}")

test2_pass = all(checks2.values())
print(f"\n{'✓ TEST PASSED' if test2_pass else '✗ TEST FAILED'}")

Path(temp_file).unlink()

print("\n" + "=" * 60)
print(
    f"SUMMARY: {'✓ ALL TESTS PASSED' if test1_pass and test2_pass else '✗ SOME TESTS FAILED'}"
)
print("=" * 60)
