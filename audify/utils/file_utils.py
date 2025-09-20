"""
Shared file and path utilities for common file operations.

This module provides utilities for path validation, directory creation,
and other file operations that are repeated across modules.
"""

import logging
import re
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class PathManager:
    """Utility class for common path and file operations."""

    @staticmethod
    def ensure_directory_exists(path: Path, create_parents: bool = True) -> Path:
        """
        Ensure that a directory exists, creating it if necessary.

        Args:
            path: Path to the directory
            create_parents: Whether to create parent directories

        Returns:
            The path (for chaining)

        Raises:
            OSError: If directory creation fails
        """
        if not path.exists():
            logger.info(f"Creating directory: {path}")
            path.mkdir(parents=create_parents, exist_ok=True)
        return path

    @staticmethod
    def validate_file_exists(file_path: Path) -> Path:
        """
        Validate that a file exists.

        Args:
            file_path: Path to the file

        Returns:
            The path (for chaining)

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    @staticmethod
    def create_temp_directory(prefix: str = "audify_") -> Path:
        """
        Create a temporary directory with a specific prefix.

        Args:
            prefix: Prefix for the temporary directory name

        Returns:
            Path to the created temporary directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        logger.debug(f"Created temporary directory: {temp_dir}")
        return temp_dir

    @staticmethod
    def setup_output_paths(
        base_dir: Path,
        title: str,
        file_extension: str = ".m4b",
    ) -> tuple[Path, Path, Path, Path]:
        """
        Set up standard output paths for audiobook creation.

        Args:
            base_dir: Base output directory
            title: Title for the audiobook (used for directory and file names)
            file_extension: Extension for the final output file

        Returns:
            Tuple of (audiobook_path, metadata_path, final_path, temp_path)
        """
        PathManager.ensure_directory_exists(base_dir)

        audiobook_path = base_dir / title
        PathManager.ensure_directory_exists(audiobook_path)

        metadata_path = audiobook_path / "chapters.txt"
        final_path = audiobook_path / f"{title}{file_extension}"
        temp_path = audiobook_path / f"{title}.tmp{file_extension}"

        logger.info(f"Output directory set to: {audiobook_path}")
        return audiobook_path, metadata_path, final_path, temp_path

    @staticmethod
    def clean_filename(filename: str) -> str:
        """
        Clean a filename by removing or replacing invalid characters.

        Args:
            filename: Original filename

        Returns:
            Cleaned filename safe for filesystem use
        """
        # Replace spaces with underscores
        cleaned = filename.lower().replace(" ", "_")

        # Replace multiple underscores with single underscore
        cleaned = re.sub(r"_+", "_", cleaned)

        # Remove leading and trailing underscores
        cleaned = cleaned.strip("_")

        # Replace accented characters with simple equivalents
        replacements = {
            "àáâãäå": "a",
            "èéêë": "e",
            "ìíîï": "i",
            "òóôõö": "o",
            "ùúûü": "u",
            "ñ": "n",
            "ç": "c",
        }

        for accented, simple in replacements.items():
            for char in accented:
                cleaned = cleaned.replace(char, simple)

        # Remove special characters, keep only alphanumeric and underscores
        cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)

        # Remove leading and trailing underscores again
        cleaned = cleaned.strip("_")

        return cleaned

    @staticmethod
    def safe_remove_file(file_path: Path, missing_ok: bool = True) -> bool:
        """
        Safely remove a file with error handling.

        Args:
            file_path: Path to the file to remove
            missing_ok: If True, don't raise error if file doesn't exist

        Returns:
            True if file was removed or didn't exist, False if removal failed
        """
        try:
            file_path.unlink(missing_ok=missing_ok)
            logger.debug(f"Removed file: {file_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to remove file {file_path}: {e}")
            return False

    @staticmethod
    def get_available_filename(directory: Path, base_name: str, extension: str) -> Path:
        """
        Get an available filename by adding a counter if the file exists.

        Args:
            directory: Directory where the file will be created
            base_name: Base name for the file
            extension: File extension (with or without leading dot)

        Returns:
            Path to an available filename
        """
        if extension and not extension.startswith("."):
            extension = f".{extension}"

        counter = 0
        while True:
            if counter == 0:
                filename = f"{base_name}{extension}"
            else:
                filename = f"{base_name}_{counter}{extension}"

            file_path = directory / filename
            if not file_path.exists():
                return file_path

            counter += 1
            if counter > 1000:  # Safety limit
                raise ValueError("Could not find available filename")
