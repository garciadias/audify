"""
API Key Management Module

This module handles reading and managing API keys from a .keys file in the project root.
The .keys file should follow the format:
API_NAME=api_key_here
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages API keys for commercial API services."""

    def __init__(self, keys_file: Optional[str | Path] = None):
        """
        Initialize API key manager.

        Args:
            keys_file: Path to the .keys file. If None, looks for
                .keys in the project root.
        """
        if keys_file is None:
            # Look for .keys file in project root
            project_root = Path(__file__).parents[2].resolve()
            keys_file = project_root / ".keys"

        self.keys_file = Path(keys_file)
        self._keys: Dict[str, str] = {}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from the .keys file."""
        if not self.keys_file.exists():
            logger.warning(f"API keys file not found: {self.keys_file}")
            logger.info(
                "Create a .keys file in the project root with format: "
                "API_NAME=api_key"
            )
            return

        try:
            with open(self.keys_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue

                    # Parse KEY=value format
                    if '=' not in line:
                        logger.warning(
                            f"Invalid format in .keys file at line "
                            f"{line_num}: {line}"
                        )
                        continue

                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    if key and value:
                        self._keys[key.upper()] = value

            logger.info(f"Loaded {len(self._keys)} API keys from {self.keys_file}")

        except Exception as e:
            logger.error(f"Error loading API keys from {self.keys_file}: {e}")

    def get_key(self, api_name: str) -> Optional[str]:
        """
        Get API key for a specific service.

        Args:
            api_name: Name of the API service (case-insensitive)

        Returns:
            API key if found, None otherwise
        """
        # Try environment variable first
        env_key = os.getenv(f"{api_name.upper()}_API_KEY")
        if env_key:
            return env_key

        # Fall back to .keys file
        return self._keys.get(api_name.upper())

    def has_key(self, api_name: str) -> bool:
        """
        Check if API key exists for a service.

        Args:
            api_name: Name of the API service (case-insensitive)

        Returns:
            True if key exists, False otherwise
        """
        return self.get_key(api_name) is not None

    def list_available_keys(self) -> list[str]:
        """
        Get list of available API service names.

        Returns:
            List of API service names that have keys configured
        """
        return list(self._keys.keys())


# Global instance
_key_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """Get or create global API key manager instance."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


def get_api_key(api_name: str) -> Optional[str]:
    """
    Convenience function to get an API key.

    Args:
        api_name: Name of the API service

    Returns:
        API key if found, None otherwise
    """
    return get_key_manager().get_key(api_name)
