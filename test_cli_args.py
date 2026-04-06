#!/usr/bin/env python3
"""Quick test to verify CLI argument handling after removing @click.argument."""

from click.testing import CliRunner

from audify.cli import cli

runner = CliRunner()

# Test 1: List tasks (subcommand)
print("\n=== Test 1: list-tasks subcommand ===")
result = runner.invoke(cli, ["list-tasks"])
print(f"Exit code: {result.exit_code}")
print(f"Output (first 200 chars): {result.output[:200]}")

# Test 2: Validate prompt (subcommand)
print("\n=== Test 2: validate-prompt subcommand ===")
result = runner.invoke(cli, ["validate-prompt", "/tmp/test.txt"])
print(f"Exit code: {result.exit_code}")
print(f"Output (first 200 chars): {result.output[:200]}")

# Test 3: List languages (flag)
print("\n=== Test 3: --list-languages flag ===")
result = runner.invoke(cli, ["--list-languages"])
print(f"Exit code: {result.exit_code}")
print(f"Output contains 'en': {'en' in result.output}")

# Test 4: File path as positional arg
print("\n=== Test 4: File path as positional ===")
result = runner.invoke(cli, ["/tmp/test.epub", "--task", "direct"])
print(f"Exit code: {result.exit_code}")
if result.exception:
    print(f"Exception: {type(result.exception).__name__}: {result.exception}")
else:
    print(f"Output (first 200 chars): {result.output[:200]}")

# Test 5: No args (should show help)
print("\n=== Test 5: No arguments ===")
result = runner.invoke(cli, [])
print(f"Exit code: {result.exit_code}")
print(f"Output contains 'Usage': {'Usage' in result.output}")
