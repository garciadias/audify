"""
Comprehensive tests for the logging utilities module.

Tests cover all functions and classes including error handling, edge cases,
and different logging configurations to achieve maximum code coverage.
"""

import logging
from unittest.mock import ANY, MagicMock, patch

from audify.utils.logging_utils import (
    LoggerMixin,
    configure_module_logging,
    get_logger,
    setup_logging,
)


class TestSetupLogging:
    """Test cases for setup_logging function."""

    def test_setup_logging_default_parameters(self):
        """Test setup_logging with default parameters."""
        with (
            patch("logging.basicConfig") as mock_basic_config,
            patch("logging.getLogger") as mock_get_logger,
            patch("logging.FileHandler") as mock_file_handler,
            patch("logging.StreamHandler") as mock_stream_handler,
            patch.dict("os.environ", {}, clear=True),
        ):
            # Mock the root logger to have no handlers
            mock_root_logger = MagicMock()
            mock_root_logger.handlers = []
            mock_root_logger.level = logging.NOTSET
            mock_get_logger.return_value = mock_root_logger

            # Mock the call chain
            def mock_get_logger_func(name=None):
                if name is None or name == "":
                    return mock_root_logger
                return MagicMock()

            mock_get_logger.side_effect = mock_get_logger_func
            # Mock file handler instance
            mock_file_handler_instance = MagicMock()
            mock_file_handler.return_value = mock_file_handler_instance

            setup_logging()

            # basicConfig should NOT be called because we add handlers directly
            mock_basic_config.assert_not_called()
            # FileHandler should be created with default log file
            mock_file_handler.assert_called_once_with(ANY)
            # Check that the argument is a Path with the correct name
            from pathlib import Path

            call_arg = mock_file_handler.call_args[0][0]
            assert isinstance(call_arg, Path)
            assert str(call_arg) == "audify.log"
            # StreamHandler should NOT be created because AUDIFY_VERBOSE not set
            mock_stream_handler.assert_not_called()
            # Ensure root logger level is set to INFO
            mock_root_logger.setLevel.assert_called_once_with(logging.INFO)
            # Ensure file handler is added to root logger
            mock_root_logger.addHandler.assert_called_once_with(
                mock_file_handler_instance
            )

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom logging level."""
        with (
            patch("logging.basicConfig") as mock_basic_config,
            patch("logging.getLogger") as mock_get_logger,
            patch("logging.FileHandler") as mock_file_handler,
            patch.dict("os.environ", {}, clear=True),
        ):
            # Mock the root logger to have no handlers
            mock_root_logger = MagicMock()
            mock_root_logger.handlers = []
            mock_root_logger.level = logging.NOTSET
            mock_get_logger.return_value = mock_root_logger
            mock_file_handler_instance = MagicMock()
            mock_file_handler.return_value = mock_file_handler_instance

            setup_logging(level=logging.DEBUG)

            mock_basic_config.assert_not_called()
            mock_root_logger.setLevel.assert_called_once_with(logging.DEBUG)
            mock_file_handler.assert_called_once_with(ANY)
            # Check that the argument is a Path with the correct name
            from pathlib import Path

            call_arg = mock_file_handler.call_args[0][0]
            assert isinstance(call_arg, Path)
            assert str(call_arg) == "audify.log"
            mock_root_logger.addHandler.assert_called_once_with(
                mock_file_handler_instance
            )

    def test_setup_logging_custom_format(self):
        """Test setup_logging with custom format string."""
        custom_format = "%(levelname)s: %(message)s"
        with (
            patch("logging.basicConfig") as mock_basic_config,
            patch("logging.getLogger") as mock_get_logger,
            patch("logging.FileHandler") as mock_file_handler,
            patch("logging.Formatter") as mock_formatter,
            patch.dict("os.environ", {}, clear=True),
        ):
            # Mock the root logger to have no handlers
            mock_root_logger = MagicMock()
            mock_root_logger.handlers = []
            mock_root_logger.level = logging.NOTSET
            mock_get_logger.return_value = mock_root_logger
            mock_file_handler_instance = MagicMock()
            mock_file_handler.return_value = mock_file_handler_instance
            mock_formatter_instance = MagicMock()
            mock_formatter.return_value = mock_formatter_instance

            setup_logging(format_string=custom_format)

            mock_basic_config.assert_not_called()
            # Formatter should be created with custom format
            mock_formatter.assert_called_once_with(
                custom_format, datefmt="%Y-%m-%d %H:%M:%S"
            )
            mock_file_handler_instance.setFormatter.assert_called_once_with(
                mock_formatter_instance
            )
            mock_root_logger.addHandler.assert_called_once_with(
                mock_file_handler_instance
            )

    def test_setup_logging_custom_module_name(self):
        """Test setup_logging with custom module name."""
        module_name = "test.module"
        with (
            patch("logging.basicConfig"),
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            logger = setup_logging(module_name=module_name)

            mock_get_logger.assert_called_with(module_name)
            assert logger == mock_logger

    def test_setup_logging_already_configured(self):
        """Test setup_logging when logging is already configured."""
        with (
            patch("logging.basicConfig") as mock_basic_config,
            patch("logging.getLogger") as mock_get_logger,
        ):
            # Mock root logger with existing handlers
            mock_root_logger = MagicMock()
            mock_root_logger.handlers = [MagicMock()]  # Has handlers
            mock_get_logger.return_value = mock_root_logger

            setup_logging()

            # basicConfig should not be called when handlers exist
            mock_basic_config.assert_not_called()

    def test_setup_logging_caller_detection_success(self):
        """Test setup_logging caller detection when frame is available."""
        with (
            patch("logging.basicConfig"),
            patch("logging.getLogger") as mock_get_logger,
            patch("inspect.currentframe") as mock_frame,
        ):
            # Mock frame structure
            mock_back_frame = MagicMock()
            mock_back_frame.f_globals = {"__name__": "test.caller.module"}
            mock_current_frame = MagicMock()
            mock_current_frame.f_back = mock_back_frame
            mock_frame.return_value = mock_current_frame

            setup_logging()

            mock_get_logger.assert_called_with("test.caller.module")

    def test_setup_logging_caller_detection_no_frame(self):
        """Test setup_logging caller detection when no frame is available."""
        with (
            patch("logging.basicConfig"),
            patch("logging.getLogger") as mock_get_logger,
            patch("inspect.currentframe") as mock_frame,
        ):
            mock_frame.return_value = None

            setup_logging()

            mock_get_logger.assert_called_with("audify")

    def test_setup_logging_caller_detection_no_back_frame(self):
        """Test setup_logging caller detection when no back frame is available."""
        with (
            patch("logging.basicConfig"),
            patch("logging.getLogger") as mock_get_logger,
            patch("inspect.currentframe") as mock_frame,
        ):
            mock_current_frame = MagicMock()
            mock_current_frame.f_back = None
            mock_frame.return_value = mock_current_frame

            setup_logging()

            mock_get_logger.assert_called_with("audify")


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_with_name(self):
        """Test get_logger with explicit name."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            logger = get_logger("test.module")

            mock_get_logger.assert_called_once_with("test.module")
            assert logger == mock_logger

    def test_get_logger_caller_detection_success(self):
        """Test get_logger caller detection when frame is available."""
        with (
            patch("logging.getLogger") as mock_get_logger,
            patch("inspect.currentframe") as mock_frame,
        ):
            # Mock frame structure
            mock_back_frame = MagicMock()
            mock_back_frame.f_globals = {"__name__": "calling.module"}
            mock_current_frame = MagicMock()
            mock_current_frame.f_back = mock_back_frame
            mock_frame.return_value = mock_current_frame

            get_logger()

            mock_get_logger.assert_called_once_with("calling.module")

    def test_get_logger_caller_detection_no_frame(self):
        """Test get_logger caller detection when no frame is available."""
        with (
            patch("logging.getLogger") as mock_get_logger,
            patch("inspect.currentframe") as mock_frame,
        ):
            mock_frame.return_value = None

            get_logger()

            mock_get_logger.assert_called_once_with("audify")

    def test_get_logger_caller_detection_no_back_frame(self):
        """Test get_logger caller detection when no back frame is available."""
        with (
            patch("logging.getLogger") as mock_get_logger,
            patch("inspect.currentframe") as mock_frame,
        ):
            mock_current_frame = MagicMock()
            mock_current_frame.f_back = None
            mock_frame.return_value = mock_current_frame

            get_logger()

            mock_get_logger.assert_called_once_with("audify")

    def test_get_logger_caller_detection_no_name_in_globals(self):
        """Test get_logger when __name__ is not in caller's globals."""
        with (
            patch("logging.getLogger") as mock_get_logger,
            patch("inspect.currentframe") as mock_frame,
        ):
            # Mock frame structure without __name__
            mock_back_frame = MagicMock()
            mock_back_frame.f_globals = {}  # No __name__ key
            mock_current_frame = MagicMock()
            mock_current_frame.f_back = mock_back_frame
            mock_frame.return_value = mock_current_frame

            get_logger()

            mock_get_logger.assert_called_once_with("audify")


class TestConfigureModuleLogging:
    """Test cases for configure_module_logging function."""

    def test_configure_module_logging_default_parameters(self):
        """Test configure_module_logging with default parameters."""
        with patch("audify.utils.logging_utils.setup_logging") as mock_setup_logging:
            mock_logger = MagicMock()
            mock_setup_logging.return_value = mock_logger

            logger = configure_module_logging("test.module")

            mock_setup_logging.assert_called_once_with(
                logging.INFO, None, "test.module"
            )
            assert logger == mock_logger

    def test_configure_module_logging_custom_parameters(self):
        """Test configure_module_logging with custom parameters."""
        custom_format = "%(name)s - %(message)s"
        with patch("audify.utils.logging_utils.setup_logging") as mock_setup_logging:
            mock_logger = MagicMock()
            mock_setup_logging.return_value = mock_logger

            logger = configure_module_logging(
                "test.module", level=logging.DEBUG, format_string=custom_format
            )

            mock_setup_logging.assert_called_once_with(
                logging.DEBUG, custom_format, "test.module"
            )
            assert logger == mock_logger


class TestLoggerMixin:
    """Test cases for LoggerMixin class."""

    def test_logger_mixin_initialization(self):
        """Test LoggerMixin initialization."""

        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        assert instance._logger is None

    def test_logger_mixin_logger_property_first_access(self):
        """Test LoggerMixin logger property on first access."""

        class TestClass(LoggerMixin):
            pass

        with patch("audify.utils.logging_utils.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            instance = TestClass()
            logger = instance.logger

            expected_name = f"{TestClass.__module__}.{TestClass.__name__}"
            mock_get_logger.assert_called_once_with(expected_name)
            assert logger == mock_logger
            assert instance._logger == mock_logger

    def test_logger_mixin_logger_property_cached(self):
        """Test LoggerMixin logger property caching."""

        class TestClass(LoggerMixin):
            pass

        with patch("audify.utils.logging_utils.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            instance = TestClass()
            # Access logger twice
            logger1 = instance.logger
            logger2 = instance.logger

            # get_logger should only be called once due to caching
            mock_get_logger.assert_called_once()
            assert logger1 == logger2 == mock_logger

    def test_logger_mixin_with_inheritance(self):
        """Test LoggerMixin with class inheritance."""

        class BaseClass(LoggerMixin):
            def __init__(self):
                super().__init__()

        class DerivedClass(BaseClass):
            def __init__(self):
                super().__init__()

        instance = DerivedClass()
        assert instance._logger is None

        with patch("audify.utils.logging_utils.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            logger = instance.logger

            expected_name = f"{DerivedClass.__module__}.{DerivedClass.__name__}"
            mock_get_logger.assert_called_once_with(expected_name)
            assert logger == mock_logger

    def test_logger_mixin_with_multiple_inheritance(self):
        """Test LoggerMixin with multiple inheritance."""

        class OtherClass:
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class TestClass(LoggerMixin, OtherClass):
            def __init__(self):
                super().__init__()

        instance = TestClass()
        assert hasattr(instance, "_logger")
        assert instance._logger is None

    def test_logger_mixin_logger_name_format(self):
        """Test LoggerMixin logger name formatting."""

        class MyTestClass(LoggerMixin):
            pass

        with patch("audify.utils.logging_utils.get_logger") as mock_get_logger:
            instance = MyTestClass()
            instance.logger  # Access to trigger creation

            expected_name = f"{MyTestClass.__module__}.MyTestClass"
            mock_get_logger.assert_called_once_with(expected_name)
