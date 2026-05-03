# tests/utils/test_combine_covers.py
import tempfile
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from PIL import Image

from audify.utils.combine_covers import combine_covers


@pytest.fixture
def mock_images():
    """Create mock PIL Image objects with different dimensions."""
    mock_img1 = Mock(spec=Image.Image)
    mock_img1.width = 300
    mock_img1.height = 400
    mock_img1.resize.return_value = mock_img1
    mock_img1.crop.return_value = mock_img1

    mock_img2 = Mock(spec=Image.Image)
    mock_img2.width = 400
    mock_img2.height = 600
    mock_img2.resize.return_value = mock_img2
    mock_img2.crop.return_value = mock_img2

    mock_img3 = Mock(spec=Image.Image)
    mock_img3.width = 350
    mock_img3.height = 500
    mock_img3.resize.return_value = mock_img3
    mock_img3.crop.return_value = mock_img3

    return [mock_img1, mock_img2, mock_img3]


@pytest.fixture
def mock_image_paths():
    """Create mock Path objects with different modification times."""
    paths = []
    for i in range(3):
        mock_path = Mock(spec=Path)
        mock_stat = Mock()
        mock_stat.st_mtime = 1000 + i  # Different modification times
        mock_path.stat.return_value = mock_stat
        paths.append(mock_path)
    return paths


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
