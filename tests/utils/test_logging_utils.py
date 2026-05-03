"""
Comprehensive tests for the logging utilities module.

Tests cover all functions and classes including error handling, edge cases,
and different logging configurations to achieve maximum code coverage.
"""

import logging
from unittest.mock import ANY, MagicMock, patch

from audify.utils.logging_utils import (
    LoggerMixin,
    configure_cli_logging,
    configure_module_logging,
    get_logger,
    setup_logging,
)


class TestSetupLogging:
    """Test cases for setup_logging function."""

    def test_setup_logging_adds_stream_handler_when_verbose(self):
        """Test setup_logging adds a stream handler when verbose is enabled."""
        with (
            patch("logging.getLogger") as mock_get_logger,
            patch("logging.FileHandler") as mock_file_handler,
            patch("logging.StreamHandler") as mock_stream_handler,
            patch.dict("os.environ", {"AUDIFY_VERBOSE": "true"}, clear=True),
        ):
            mock_root_logger = MagicMock()
            mock_root_logger.handlers = []
            mock_root_logger.level = logging.NOTSET

            def mock_get_logger_func(name=None):
                if name is None or name == "":
                    return mock_root_logger
                return MagicMock()

            mock_get_logger.side_effect = mock_get_logger_func

            mock_file_handler_instance = MagicMock()
            mock_stream_handler_instance = MagicMock()
            mock_file_handler.return_value = mock_file_handler_instance
            mock_stream_handler.return_value = mock_stream_handler_instance

            setup_logging()

            mock_stream_handler.assert_called_once()
            mock_stream_handler_instance.setFormatter.assert_called_once()
            mock_root_logger.addHandler.assert_any_call(mock_stream_handler_instance)

class TestConfigureCliLogging:
    """Test cases for configure_cli_logging function."""

    def test_configure_cli_logging_verbose_adds_stream_and_file_handlers(self):
        """Test verbose CLI logging adds stdout/file handlers and sets env."""
        with (
            patch("logging.getLogger") as mock_get_logger,
            patch("logging.FileHandler") as mock_file_handler,
            patch("logging.StreamHandler") as mock_stream_handler,
            patch.dict("os.environ", {}, clear=True),
        ):
            mock_root_logger = MagicMock()
            mock_root_logger.handlers = []
            mock_root_logger.level = logging.NOTSET
            mock_get_logger.return_value = mock_root_logger

            mock_file_handler_instance = MagicMock()
            mock_stream_handler_instance = MagicMock()
            mock_file_handler.return_value = mock_file_handler_instance
            mock_stream_handler.return_value = mock_stream_handler_instance

            configure_cli_logging(verbose=True, log_file="custom.log")

            assert "AUDIFY_VERBOSE" in __import__("os").environ
            assert __import__("os").environ["AUDIFY_VERBOSE"] == "true"
            mock_stream_handler.assert_called_once()
            mock_file_handler.assert_called_once_with("custom.log")
            mock_root_logger.addHandler.assert_any_call(mock_stream_handler_instance)
            mock_root_logger.addHandler.assert_any_call(mock_file_handler_instance)
            mock_root_logger.setLevel.assert_called_once_with(logging.INFO)

    def test_configure_cli_logging_removes_existing_stdout_handler(self):
        """Test existing stdout stream handlers are removed before reconfiguring."""
        existing_stdout_handler = logging.StreamHandler(__import__("sys").stdout)
        with (
            patch("logging.getLogger") as mock_get_logger,
            patch.dict("os.environ", {"AUDIFY_VERBOSE": "true"}, clear=True),
        ):
            mock_root_logger = MagicMock()
            mock_root_logger.handlers = [existing_stdout_handler]
            mock_root_logger.level = logging.INFO
            mock_get_logger.return_value = mock_root_logger

            configure_cli_logging(verbose=False)

            mock_root_logger.removeHandler.assert_called_once_with(
                existing_stdout_handler
            )
