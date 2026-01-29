#!/usr/bin/env python3
"""Tests for uncovered paths in api_config and api_keys."""

import os
import tempfile
from unittest.mock import patch

from audify.utils.api_config import CommercialAPIConfig
from audify.utils.api_keys import APIKeyManager


class TestCommercialAPIConfigNoKey:
    """Test CommercialAPIConfig when no API key is found."""

    def test_no_api_key_warning(self):
        """Test that missing API key results in None."""
        with patch(
            "audify.utils.api_keys.get_api_key",
            return_value=None,
        ):
            config = CommercialAPIConfig(model="deepseek-chat")
            assert config.api_key is None


class TestAPIKeyManagerLoadError:
    """Test APIKeyManager when keys file has load errors."""

    def test_load_keys_nonexistent_file(self):
        """Test loading from nonexistent file."""
        manager = APIKeyManager(keys_file="/nonexistent/.keys")
        assert manager.get_key("DEEPSEEK") is None

    def test_load_keys_permission_error(self):
        """Test permission error reading keys file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".keys", delete=False
        ) as f:
            f.write("DEEPSEEK=test-key\n")
            f.flush()
            path = f.name

        os.chmod(path, 0o000)
        try:
            manager = APIKeyManager(keys_file=path)
            assert manager.get_key("DEEPSEEK") is None
        finally:
            os.chmod(path, 0o644)
            os.unlink(path)
