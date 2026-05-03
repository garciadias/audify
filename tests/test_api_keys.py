"""Tests for API key management functionality."""

import os
import tempfile
from pathlib import Path

from audify.utils.api_keys import APIKeyManager, get_api_key, get_key_manager


class TestAPIKeyManager:
    """Test cases for APIKeyManager class."""

    def test_has_key(self):
        """Test has_key method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.keys', delete=False) as f:
            f.write("DEEPSEEK=test-key-123\n")
            keys_file = f.name

        try:
            manager = APIKeyManager(keys_file)
            assert manager.has_key('deepseek')
            assert not manager.has_key('nonexistent')
        finally:
            os.unlink(keys_file)

    def test_list_available_keys(self):
        """Test listing available API keys."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.keys', delete=False) as f:
            f.write("DEEPSEEK=test-key-123\n")
            f.write("ANTHROPIC=test-key-456\n")
            f.write("OPENAI=test-key-789\n")
            keys_file = f.name

        try:
            manager = APIKeyManager(keys_file)
            keys = manager.list_available_keys()
            assert len(keys) == 3
            assert 'DEEPSEEK' in keys
            assert 'ANTHROPIC' in keys
            assert 'OPENAI' in keys
        finally:
            os.unlink(keys_file)


    def test_invalid_line_format(self):
        """Test handling of invalid line formats."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.keys', delete=False) as f:
            f.write("VALID_KEY=valid-value\n")
            f.write("INVALID_LINE_NO_EQUALS\n")
            f.write("=value_without_key\n")
            f.write("key_without_value=\n")
            keys_file = f.name

        try:
            manager = APIKeyManager(keys_file)
            # Only valid key should be loaded
            assert manager.get_key('VALID_KEY') == 'valid-value'
            assert len(manager._keys) == 1
        finally:
            os.unlink(keys_file)
