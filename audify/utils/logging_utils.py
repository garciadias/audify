"""Shared logging utilities for audify."""

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
    """Set up logging and return a logger for the calling module.

    Adds a file handler writing to audify.log, and a stdout handler when
    AUDIFY_VERBOSE=1 is set. Safe to call multiple times.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    root_logger = logging.getLogger()

    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        file_handler = logging.FileHandler(Path("audify.log"))
        file_handler.setFormatter(
            logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        )
        root_logger.addHandler(file_handler)

    verbose = os.environ.get("AUDIFY_VERBOSE", "").lower() in ("1", "true", "yes")
    stream_handlers = [
        h
        for h in root_logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    if verbose and not stream_handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(stream_handler)

    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(level)

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
    """Configure logging for CLI entry points.

    Always writes to a log file. Adds stdout output when verbose=True.
    """
    root_logger = logging.getLogger()

    if verbose:
        os.environ["AUDIFY_VERBOSE"] = "true"
    else:
        os.environ.pop("AUDIFY_VERBOSE", None)

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
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(stream_handler)

    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        path = log_file or "audify.log"
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger for the given name, or the caller's module name."""
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
    """Configure logging for a specific module. Delegates to setup_logging."""
    return setup_logging(level, format_string, module_name)


class LoggerMixin:
    """Mixin that adds a lazy ``self.logger`` property to any class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = get_logger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger
