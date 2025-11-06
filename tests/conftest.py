import pytest

from tests.fixtures.readers import *  # noqa: F403

pytest_plugins = ["tests.fixtures.synthesizers"]


# Make test file fixtures available globally
@pytest.fixture(scope="session", autouse=True)
def setup_test_files(test_pdf_file, test_epub_file):
    """Auto-setup test files for all tests that need them."""
    # Files are created by the individual fixtures
    # This just ensures they're available when needed
    pass
