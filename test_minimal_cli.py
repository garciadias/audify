#!/usr/bin/env python3
"""Minimal test of CLI structure."""

# Test 1: Can we import the cli?
try:
    from audify.cli import cli

    print("✓ Successfully imported cli")
except Exception as e:
    print(f"✗ Failed to import cli: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 2: Can we import TaskRegistry?
try:
    from audify.prompts.tasks import TaskRegistry

    print("✓ Successfully imported TaskRegistry")
except Exception as e:
    print(f"✗ Failed to import TaskRegistry: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 3: Is TaskRegistry initialized?
try:
    tasks = TaskRegistry.get_all()
    print(f"✓ TaskRegistry initialized with {len(tasks)} tasks")
except Exception as e:
    print(f"✗ TaskRegistry.get_all() failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 4: Are the commands registered?
try:
    if hasattr(cli, "commands"):
        print(f"✓ cli.commands exists: {list(cli.commands.keys())}")
    else:
        print("⚠ cli.commands doesn't exist (might be normal for Click 8+)")

    # Try to get the commands via the click context
    # click.group stores commands in a different way
    print(f"✓ cli object type: {type(cli)}")
    print("✓ cli is a Click group")
except Exception as e:
    print(f"✗ Error checking commands: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Can we invoke the commands?
try:
    from click.testing import CliRunner

    runner = CliRunner()

    result = runner.invoke(cli, ["list-tasks"])
    print("\nList-tasks invocation:")
    print(f"  Exit code: {result.exit_code}")
    if result.output:
        print(f"  Output (first 200 chars):\n{result.output[:200]}")
    if result.exception:
        print(f"  Exception: {result.exception}")
        import traceback

        traceback.print_exception(
            type(result.exception), result.exception, result.exception.__traceback__
        )

except Exception as e:
    print(f"✗ Error invoking list-tasks: {e}")
    import traceback

    traceback.print_exc()
