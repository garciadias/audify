"""
Shared logging utilities for consistent logging configuration across modules.

This module provides standardized logging setup to reduce code duplication
and ensure consistent logging behavior throughout the application.
"""

import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    module_name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration and return a logger for the calling module.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (uses default if None)
        module_name: Name for the logger (uses caller's __name__ if None)

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    root_logger = logging.getLogger()

    # Add file handler if none exists
    file_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
    ]
    if not file_handlers:
        log_file = Path("audify.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Add stream handler only if verbose environment variable is set
    # and no stream handler exists
    stream_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    verbose = os.environ.get("AUDIFY_VERBOSE", "").lower() in ("1", "true", "yes")
    if verbose and not stream_handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(stream_handler)

    # Set level if not set
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(level)

    # Return logger for the specific module
    if module_name is None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            module_name = frame.f_back.f_globals.get("__name__", "audify")
        else:
            module_name = "audify"

    return logging.getLogger(module_name)


def configure_cli_logging(
    verbose: bool = False, log_file: Optional[str] = None
) -> None:
    """
    Configure logging for CLI commands.

    Sets up a file handler for all logs and optionally a stream handler
    for verbose output.
    This should be called early in CLI entry points.

    Args:
        verbose: If True, add a stream handler to stdout.
        log_file: Path to log file (default: 'audify.log' in current directory).
    """
    root_logger = logging.getLogger()

    # Set environment variable for child processes and other modules
    if verbose:
        os.environ["AUDIFY_VERBOSE"] = "true"
    else:
        os.environ.pop("AUDIFY_VERBOSE", None)

    # Remove existing stdout handlers
    stdout_handlers = [
        h
        for h in root_logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and hasattr(h, "stream")
        and h.stream is sys.stdout
    ]
    for handler in stdout_handlers:
        root_logger.removeHandler(handler)

    if verbose:
        # Add a stdout stream handler with simple formatter
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(stream_handler)

    # Ensure file handler exists
    file_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
    ]
    if not file_handlers:
        if log_file is None:
            log_file = "audify.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set level if not set
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name, or the caller's module name.

    Args:
        name: Logger name (uses caller's __name__ if None)

    Returns:
        Logger instance
    """
    if name is None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "audify")
        else:
            name = "audify"

    return logging.getLogger(name)


def configure_module_logging(
    module_name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for a specific module.

    Args:
        module_name: Name of the module
        level: Logging level
        format_string: Custom format string

    Returns:
        Configured logger for the module
    """
    return setup_logging(level, format_string, module_name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.

    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                super().__init__()
                self.logger.info("MyClass initialized")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if self._logger is None:
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = get_logger(logger_name)
        return self._logger
