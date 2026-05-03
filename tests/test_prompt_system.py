#!/usr/bin/env python3
"""Tests for the flexible prompt system (PromptManager, TaskRegistry, TaskConfig)."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from audify.prompts.manager import PromptManager
from audify.prompts.tasks import TaskConfig, TaskRegistry

class TestPromptManager:
    """Test cases for PromptManager."""

    def test_load_prompt_file_not_found(self):
        manager = PromptManager()
        with pytest.raises(FileNotFoundError):
            manager.load_prompt_file("/nonexistent/path/prompt.txt")

    def test_load_prompt_file_empty(self):
        manager = PromptManager()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            with pytest.raises(ValueError, match="empty"):
                manager.load_prompt_file(f.name)
        Path(f.name).unlink()


    def test_get_prompt_direct_task(self):
        manager = PromptManager()
        prompt = manager.get_prompt(task="direct")
        assert prompt == ""


    def test_get_prompt_unknown_task(self):
        manager = PromptManager()
        with pytest.raises(ValueError, match="Unknown task"):
            manager.get_prompt(task="nonexistent_task_xyz")

    def test_validate_prompt_empty(self):
        manager = PromptManager()
        is_valid, msg = manager.validate_prompt("")
        assert is_valid is False
        assert "empty" in msg.lower()
