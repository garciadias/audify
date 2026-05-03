from pathlib import Path
from unittest.mock import patch

import pytest

from audify.utils.file_utils import PathManager


class TestPathManager:
    """Test cases for PathManager class methods."""

    def test_validate_file_exists_valid_file(self, tmp_path):
        """Test validating an existing file."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        result = PathManager.validate_file_exists(test_file)

        assert result == test_file

    def test_validate_file_exists_missing_file(self, tmp_path):
        """Test validating a missing file."""
        missing_file = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError, match="File not found"):
            PathManager.validate_file_exists(missing_file)

class TestPathManagerGetAvailablefile_name:
    """Test cases for PathManager.get_available_file_name method."""


    def test_get_available_file_name_extension_without_dot(self, tmp_path):
        """Test get_available_file_name with extension without leading dot."""
        directory = tmp_path
        base_name = "test_file"
        extension = "txt"

        result = PathManager.get_available_file_name(directory, base_name, extension)

        expected = directory / "test_file.txt"
        assert result == expected

    @patch("pathlib.Path.exists")
    def test_get_available_file_name_safety_limit_reached(self, mock_exists, tmp_path):
        """Test get_available_file_name raises ValueError when safety limit is reached.
        """
        directory = tmp_path
        base_name = "test_file"
        extension = ".txt"

        # Mock exists to always return True (file always exists)
        mock_exists.return_value = True

        with pytest.raises(ValueError, match="Could not find available file_name"):
            PathManager.get_available_file_name(directory, base_name, extension)
