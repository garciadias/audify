"""
Shared logging utilities for consistent logging configuration across modules.

This module provides standardized logging setup to reduce code duplication
and ensure consistent logging behavior throughout the application.
"""

import logging
import sys
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

    # Configure basic logging only if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format=format_string,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    # Return logger for the specific module
    if module_name is None:
        # Try to get the caller's module name
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            module_name = frame.f_back.f_globals.get("__name__", "audify")
        else:
            module_name = "audify"

    return logging.getLogger(module_name)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name, or the caller's module name.

    Args:
        name: Logger name (uses caller's __name__ if None)

    Returns:
        Logger instance
    """
    if name is None:
        # Try to get the caller's module name
        import inspect

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
