#!/usr/bin/env python3
"""Test imports and basic functionality."""

import sys
import traceback

try:
    print("Importing audify.cli...")
    from audify.cli import cli
    print("✓ cli imported successfully")
except Exception as e:
    print(f"✗ Failed to import cli: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing audify.audiobook_creator...")
    from audify.audiobook_creator import AudiobookCreator, DirectoryAudiobookCreator, LLMClient
    print("✓ audiobook_creator imported successfully")
except Exception as e:
    print(f"✗ Failed to import audiobook_creator: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Checking CLI structure...")
    print(f"  cli type: {type(cli)}")
    print(f"  cli is callable: {callable(cli)}")
    if hasattr(cli, 'commands'):
        print(f"  cli.commands: {list(cli.commands.keys())}")
except Exception as e:
    print(f"✗ Failed to check CLI structure: {e}")
    traceback.print_exc()

print("\n✓ All imports successful!")
