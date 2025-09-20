from pathlib import Path
from unittest.mock import patch

import pytest

from audify.utils.file_utils import PathManager


class TestPathManager:
    """Test cases for PathManager class methods."""

    def test_ensure_directory_exists_new_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "new_directory"

        result = PathManager.ensure_directory_exists(new_dir)

        assert new_dir.exists()
        assert result == new_dir

    def test_ensure_directory_exists_existing_directory(self, tmp_path):
        """Test with an already existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = PathManager.ensure_directory_exists(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir

    def test_ensure_directory_exists_without_parents(self, tmp_path):
        """Test creating directory without creating parents."""
        new_dir = tmp_path / "parent" / "child"

        with pytest.raises(FileNotFoundError):
            PathManager.ensure_directory_exists(new_dir, create_parents=False)

    def test_ensure_directory_exists_with_parents(self, tmp_path):
        """Test creating directory with parent directories."""
        new_dir = tmp_path / "parent" / "child"

        result = PathManager.ensure_directory_exists(new_dir, create_parents=True)

        assert new_dir.exists()
        assert result == new_dir

    def test_ensure_directory_exists_permission_error(self, tmp_path):
        """Test handling permission errors during directory creation."""
        new_dir = tmp_path / "restricted"

        with patch.object(Path, "mkdir") as mock_mkdir:
            mock_mkdir.side_effect = OSError("Permission denied")

            with pytest.raises(OSError, match="Permission denied"):
                PathManager.ensure_directory_exists(new_dir)

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

    def test_create_temp_directory_default_prefix(self):
        """Test creating temporary directory with default prefix."""
        with patch("tempfile.mkdtemp") as mock_mkdtemp:
            mock_mkdtemp.return_value = "/tmp/audify_test123"

            result = PathManager.create_temp_directory()

            assert result == Path("/tmp/audify_test123")
            mock_mkdtemp.assert_called_once_with(prefix="audify_")

    def test_create_temp_directory_custom_prefix(self):
        """Test creating temporary directory with custom prefix."""
        with patch("tempfile.mkdtemp") as mock_mkdtemp:
            mock_mkdtemp.return_value = "/tmp/custom_test456"

            result = PathManager.create_temp_directory(prefix="custom_")

            assert result == Path("/tmp/custom_test456")
            mock_mkdtemp.assert_called_once_with(prefix="custom_")

    def test_create_temp_directory_error(self):
        """Test handling errors during temp directory creation."""
        with patch("tempfile.mkdtemp") as mock_mkdtemp:
            mock_mkdtemp.side_effect = OSError("Cannot create temp directory")

            with pytest.raises(OSError, match="Cannot create temp directory"):
                PathManager.create_temp_directory()

    def test_setup_output_paths_new_directory(self, tmp_path):
        """Test setting up output paths with new directory."""
        base_dir = tmp_path / "output"
        title = "test_book"

        audiobook_path, _, final_path, temp_path = PathManager.setup_output_paths(
            base_dir, title
        )

        assert base_dir.exists()
        assert audiobook_path.exists()
        assert audiobook_path == base_dir / title
        assert final_path == audiobook_path / f"{title}.m4b"
        assert temp_path == audiobook_path / f"{title}.tmp.m4b"

    def test_setup_output_paths_custom_extension(self, tmp_path):
        """Test setting up output paths with custom file extension."""
        base_dir = tmp_path / "output"
        title = "test_book"
        extension = ".mp3"

        audiobook_path, _, final_path, temp_path = PathManager.setup_output_paths(
            base_dir, title, file_extension=extension
        )

        assert final_path == audiobook_path / f"{title}.mp3"
        assert temp_path == audiobook_path / f"{title}.tmp.mp3"

    def test_setup_output_paths_existing_directory(self, tmp_path):
        """Test setting up output paths with existing directories."""
        base_dir = tmp_path / "output"
        base_dir.mkdir()
        title = "test_book"
        audiobook_dir = base_dir / title
        audiobook_dir.mkdir()

        audiobook_path, _, _, _ = PathManager.setup_output_paths(
            base_dir, title
        )

        assert audiobook_path.exists()
        assert audiobook_path == base_dir / title

    def test_clean_filename_basic_cleaning(self):
        """Test basic filename cleaning operations."""
        filename = "Test Book Title"

        result = PathManager.clean_filename(filename)

        assert result == "test_book_title"

    def test_clean_filename_multiple_spaces(self):
        """Test cleaning filename with multiple spaces."""
        filename = "Test   Book    Title"

        result = PathManager.clean_filename(filename)

        assert result == "test_book_title"

    def test_clean_filename_special_characters(self):
        """Test cleaning filename with special characters."""
        filename = "Test-Book@Title#2023!"

        result = PathManager.clean_filename(filename)

        assert result == "testbooktitle2023"

    def test_clean_filename_accented_characters(self):
        """Test cleaning filename with accented characters."""
        filename = "Café Español Naïve"

        result = PathManager.clean_filename(filename)

        assert result == "cafe_espanol_naive"

    def test_clean_filename_leading_trailing_underscores(self):
        """Test cleaning filename with leading/trailing underscores."""
        filename = "__Test Book__"

        result = PathManager.clean_filename(filename)

        assert result == "test_book"

    def test_clean_filename_only_special_characters(self):
        """Test cleaning filename with only special characters."""
        filename = "!@#$%^&*()"

        result = PathManager.clean_filename(filename)

        assert result == ""

    def test_clean_filename_numbers_and_letters(self):
        """Test cleaning filename with numbers and letters."""
        filename = "Book 123 Chapter 456"

        result = PathManager.clean_filename(filename)

        assert result == "book_123_chapter_456"

    def test_clean_filename_all_accented_types(self):
        """Test cleaning filename with all types of accented characters."""
        filename = "àáâãäåèéêëìíîïòóôõöùúûüñç"

        result = PathManager.clean_filename(filename)

        assert result == "aaaaaaeeeeiiiiooooouuuunc"

    def test_safe_remove_file_existing_file(self, tmp_path):
        """Test safely removing an existing file."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        result = PathManager.safe_remove_file(test_file)

        assert result is True
        assert not test_file.exists()

    def test_safe_remove_file_missing_file_ok(self, tmp_path):
        """Test safely removing a missing file with missing_ok=True."""
        missing_file = tmp_path / "missing.txt"

        result = PathManager.safe_remove_file(missing_file, missing_ok=True)

        assert result is True

    def test_safe_remove_file_missing_file_not_ok(self, tmp_path):
        """Test safely removing a missing file with missing_ok=False."""
        missing_file = tmp_path / "missing.txt"

        result = PathManager.safe_remove_file(missing_file, missing_ok=False)

        assert result is False

    def test_safe_remove_file_permission_error(self, tmp_path):
        """Test handling permission errors during file removal."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        with patch.object(Path, "unlink") as mock_unlink:
            mock_unlink.side_effect = PermissionError("Permission denied")

            result = PathManager.safe_remove_file(test_file)

            assert result is False


class TestPathManagerGetAvailableFilename:
    """Test cases for PathManager.get_available_filename method."""

    def test_get_available_filename_no_existing_file(self, tmp_path):
        """Test get_available_filename when no file exists."""
        directory = tmp_path
        base_name = "test_file"
        extension = ".txt"

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test_file.txt"
        assert result == expected

    def test_get_available_filename_extension_without_dot(self, tmp_path):
        """Test get_available_filename with extension without leading dot."""
        directory = tmp_path
        base_name = "test_file"
        extension = "txt"

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test_file.txt"
        assert result == expected

    def test_get_available_filename_extension_with_dot(self, tmp_path):
        """Test get_available_filename with extension with leading dot."""
        directory = tmp_path
        base_name = "test_file"
        extension = ".txt"

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test_file.txt"
        assert result == expected

    def test_get_available_filename_file_exists_increment_counter(self, tmp_path):
        """Test get_available_filename when original file exists."""
        directory = tmp_path
        base_name = "test_file"
        extension = ".txt"

        # Create the original file
        original_file = directory / "test_file.txt"
        original_file.touch()

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test_file_1.txt"
        assert result == expected

    def test_get_available_filename_multiple_files_exist(self, tmp_path):
        """Test get_available_filename when multiple numbered files exist."""
        directory = tmp_path
        base_name = "test_file"
        extension = ".txt"

        # Create multiple files
        (directory / "test_file.txt").touch()
        (directory / "test_file_1.txt").touch()
        (directory / "test_file_2.txt").touch()

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test_file_3.txt"
        assert result == expected

    def test_get_available_filename_empty_base_name(self, tmp_path):
        """Test get_available_filename with empty base name."""
        directory = tmp_path
        base_name = ""
        extension = ".txt"

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / ".txt"
        assert result == expected

    def test_get_available_filename_empty_extension(self, tmp_path):
        """Test get_available_filename with empty extension."""
        directory = tmp_path
        base_name = "test_file"
        extension = ""

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test_file"
        assert result == expected

    def test_get_available_filename_special_characters_in_base_name(self, tmp_path):
        """Test get_available_filename with special characters in base name."""
        directory = tmp_path
        base_name = "test-file with spaces"
        extension = ".txt"

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test-file with spaces.txt"
        assert result == expected

    def test_get_available_filename_complex_extension(self, tmp_path):
        """Test get_available_filename with complex extension."""
        directory = tmp_path
        base_name = "archive"
        extension = ".tar.gz"

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "archive.tar.gz"
        assert result == expected

    @patch("pathlib.Path.exists")
    def test_get_available_filename_safety_limit_reached(self, mock_exists, tmp_path):
        """Test get_available_filename raises ValueError when safety limit is reached.
        """
        directory = tmp_path
        base_name = "test_file"
        extension = ".txt"

        # Mock exists to always return True (file always exists)
        mock_exists.return_value = True

        with pytest.raises(ValueError, match="Could not find available filename"):
            PathManager.get_available_filename(directory, base_name, extension)

    def test_get_available_filename_counter_at_limit_minus_one(self, tmp_path):
        """Test get_available_filename works at counter 1000."""
        directory = tmp_path
        base_name = "test_file"
        extension = ".txt"

        # Create files from test_file.txt to test_file_999.txt
        # This will force the method to return test_file_1000.txt
        (directory / "test_file.txt").touch()
        for i in range(1, 1000):
            (directory / f"test_file_{i}.txt").touch()

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test_file_1000.txt"
        assert result == expected

    def test_get_available_filename_directory_path_object(self, tmp_path):
        """Test get_available_filename with Path object as directory."""
        directory = Path(tmp_path)
        base_name = "test_file"
        extension = ".txt"

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test_file.txt"
        assert result == expected
        assert isinstance(result, Path)

    def test_get_available_filename_preserves_path_type(self, tmp_path):
        """Test that get_available_filename returns Path object."""
        directory = tmp_path
        base_name = "test_file"
        extension = ".txt"

        result = PathManager.get_available_filename(directory, base_name, extension)

        assert isinstance(result, Path)

    def test_get_available_filename_counter_formatting(self, tmp_path):
        """Test that counter is properly formatted in filename."""
        directory = tmp_path
        base_name = "test"
        extension = ".txt"

        # Create files to test counter formatting
        (directory / "test.txt").touch()
        (directory / "test_1.txt").touch()
        (directory / "test_2.txt").touch()
        (directory / "test_3.txt").touch()
        (directory / "test_4.txt").touch()
        (directory / "test_5.txt").touch()
        (directory / "test_6.txt").touch()
        (directory / "test_7.txt").touch()
        (directory / "test_8.txt").touch()
        (directory / "test_9.txt").touch()

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "test_10.txt"
        assert result == expected

    def test_get_available_filename_with_numeric_base_name(self, tmp_path):
        """Test get_available_filename with numeric base name."""
        directory = tmp_path
        base_name = "123"
        extension = ".txt"

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "123.txt"
        assert result == expected

    def test_get_available_filename_unicode_characters(self, tmp_path):
        """Test get_available_filename with unicode characters in base name."""
        directory = tmp_path
        base_name = "tëst_fïlé"
        extension = ".txt"

        result = PathManager.get_available_filename(directory, base_name, extension)

        expected = directory / "tëst_fïlé.txt"
        assert result == expected
