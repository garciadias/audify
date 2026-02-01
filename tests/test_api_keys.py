"""Tests for API key management functionality."""

import os
import tempfile
from pathlib import Path

from audify.utils.api_keys import APIKeyManager, get_api_key, get_key_manager


class TestAPIKeyManager:
    """Test cases for APIKeyManager class."""

    def test_init_with_nonexistent_file(self):
        """Test initialization with a non-existent .keys file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_file = Path(tmpdir) / "nonexistent.keys"
            manager = APIKeyManager(keys_file)
            assert manager.keys_file == keys_file
            assert len(manager._keys) == 0

    def test_load_keys_from_file(self):
        """Test loading API keys from a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.keys', delete=False) as f:
            f.write("DEEPSEEK=test-key-123\n")
            f.write("ANTHROPIC=test-key-456\n")
            f.write("# This is a comment\n")
            f.write("\n")  # Empty line
            f.write("OPENAI=test-key-789\n")
            keys_file = f.name

        try:
            manager = APIKeyManager(keys_file)
            assert manager.get_key('deepseek') == 'test-key-123'
            assert manager.get_key('ANTHROPIC') == 'test-key-456'
            assert manager.get_key('openai') == 'test-key-789'
            assert len(manager._keys) == 3
        finally:
            os.unlink(keys_file)

    def test_get_key_case_insensitive(self):
        """Test that get_key is case-insensitive."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.keys', delete=False) as f:
            f.write("DEEPSEEK=test-key-123\n")
            keys_file = f.name

        try:
            manager = APIKeyManager(keys_file)
            assert manager.get_key('deepseek') == 'test-key-123'
            assert manager.get_key('DEEPSEEK') == 'test-key-123'
            assert manager.get_key('DeepSeek') == 'test-key-123'
        finally:
            os.unlink(keys_file)

    def test_get_key_nonexistent(self):
        """Test getting a non-existent API key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.keys', delete=False) as f:
            f.write("DEEPSEEK=test-key-123\n")
            keys_file = f.name

        try:
            manager = APIKeyManager(keys_file)
            assert manager.get_key('nonexistent') is None
        finally:
            os.unlink(keys_file)

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

    def test_environment_variable_override(self):
        """Test that environment variables override .keys file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.keys', delete=False) as f:
            f.write("DEEPSEEK=file-key\n")
            keys_file = f.name

        try:
            # Set environment variable
            os.environ['DEEPSEEK_API_KEY'] = 'env-key'

            manager = APIKeyManager(keys_file)
            # Environment variable should take precedence
            assert manager.get_key('deepseek') == 'env-key'

            # Clean up environment
            del os.environ['DEEPSEEK_API_KEY']
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

    def test_whitespace_handling(self):
        """Test handling of whitespace in keys file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.keys', delete=False) as f:
            f.write("  DEEPSEEK  =  test-key-123  \n")
            f.write("ANTHROPIC=test-key-456\n")
            keys_file = f.name

        try:
            manager = APIKeyManager(keys_file)
            # Whitespace should be stripped
            assert manager.get_key('DEEPSEEK') == 'test-key-123'
            assert manager.get_key('ANTHROPIC') == 'test-key-456'
        finally:
            os.unlink(keys_file)


class TestGlobalKeyManager:
    """Test cases for global key manager functions."""

    def test_get_key_manager_singleton(self):
        """Test that get_key_manager returns the same instance."""
        manager1 = get_key_manager()
        manager2 = get_key_manager()
        assert manager1 is manager2

    def test_get_api_key_convenience_function(self):
        """Test the convenience function for getting API keys."""
        # This will use the actual .keys file if it exists
        # or return None if not found
        result = get_api_key('nonexistent_key')
        assert result is None
