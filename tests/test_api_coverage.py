#!/usr/bin/env python3
"""Tests for uncovered paths in api_config and api_keys."""

import os
import tempfile
from unittest.mock import patch

from audify.utils.api_config import CommercialAPIConfig
from audify.utils.api_keys import APIKeyManager

class TestAPIKeyManagerLoadError:
    """Test APIKeyManager when keys file has load errors."""


    def test_load_keys_permission_error(self):
        """Test permission error reading keys file."""
        # Clear environment variable that would override file key
        saved_env = os.environ.pop("DEEPSEEK_API_KEY", None)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".keys", delete=False
        ) as f:
            f.write("DEEPSEEK=test-key\n")
            path = f.name

        try:
            # Patch open to raise PermissionError (chmod 0o000 doesn't block root)
            with patch("builtins.open", side_effect=PermissionError):
                manager = APIKeyManager(keys_file=path)
                assert manager.get_key("DEEPSEEK") is None
        finally:
            os.unlink(path)
            if saved_env is not None:
                os.environ["DEEPSEEK_API_KEY"] = saved_env
