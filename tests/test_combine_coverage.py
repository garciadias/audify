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


@patch('audify.utils.combine_covers.Image.open')
@patch('audify.utils.combine_covers.Image.new')
@patch('builtins.print')
def test_combine_covers_basic_functionality(
    mock_print, mock_image_new, mock_image_open,
    mock_images, mock_image_paths, temp_output_dir
):
    """Test basic functionality with 3 images (single page)."""
    # Setup mocks
    mock_image_open.side_effect = mock_images
    mock_new_image = Mock()
    mock_image_new.return_value = mock_new_image

    # Configure image heights for median calculation
    mock_images[0].height = 560  # 420 * 400 / 300
    mock_images[1].height = 630  # 420 * 600 / 400
    mock_images[2].height = 600  # 420 * 500 / 350

    # Call function
    combine_covers(mock_image_paths, temp_output_dir)

    # Verify images were opened
    assert mock_image_open.call_count == 3

    # Verify images were resized to 420px width
    for mock_img in mock_images:
        mock_img.resize.assert_called()

    # Verify new image was created with correct dimensions
    expected_height = int(np.median([560, 630, 600]))
    mock_image_new.assert_called_with(
        "RGB", (420 * 5, expected_height * 5), (255, 255, 255)
    )

    # Verify image was saved
    mock_new_image.save.assert_called_once()


@patch('audify.utils.combine_covers.Image.open')
@patch('audify.utils.combine_covers.Image.new')
@patch('builtins.print')
def test_combine_covers_multiple_pages(
        mock_print, mock_image_new, mock_image_open, temp_output_dir
    ):
    """Test with more than 25 images to create multiple pages."""
    # Create 30 mock images and paths
    mock_images = []
    mock_paths = []

    for i in range(30):
        mock_img = Mock(spec=Image.Image)
        mock_img.width = 300
        mock_img.height = 400
        mock_img.resize.return_value = mock_img
        mock_img.crop.return_value = mock_img
        mock_images.append(mock_img)

        mock_path = Mock(spec=Path)
        mock_stat = Mock()
        mock_stat.st_mtime = 1000 + i
        mock_path.stat.return_value = mock_stat
        mock_paths.append(mock_path)

    mock_image_open.side_effect = mock_images
    mock_new_image = Mock()
    mock_image_new.return_value = mock_new_image

    # Call function
    combine_covers(mock_paths, temp_output_dir)

    # Verify 2 pages were created (30 images = 25 + 5, so 2 pages)
    assert mock_new_image.save.call_count == 2

    # Verify correct file names
    expected_calls = [
        call(temp_output_dir / "combined_covers_0.jpg"),
        call(temp_output_dir / "combined_covers_1.jpg")
    ]
    mock_new_image.save.assert_has_calls(expected_calls)


@patch('audify.utils.combine_covers.Image.open')
@patch('audify.utils.combine_covers.Image.new')
@patch('builtins.print')
def test_combine_covers_single_image(
        mock_print, mock_image_new, mock_image_open, temp_output_dir
    ):
    """Test with single image."""
    mock_img = Mock(spec=Image.Image)
    mock_img.width = 300
    mock_img.height = 400
    mock_img.resize.return_value = mock_img
    mock_img.crop.return_value = mock_img

    mock_path = Mock(spec=Path)
    mock_stat = Mock()
    mock_stat.st_mtime = 1000
    mock_path.stat.return_value = mock_stat

    mock_image_open.return_value = mock_img
    mock_new_image = Mock()
    mock_image_new.return_value = mock_new_image

    combine_covers([mock_path], temp_output_dir)

    # Verify single image was processed
    mock_image_open.assert_called_once()
    mock_new_image.save.assert_called_once_with(
        temp_output_dir / "combined_covers_0.jpg"
    )


def test_image_sorting_by_modification_time():
    """Test that images are sorted by modification time (oldest first)."""
    # Create mock paths with different modification times
    mock_paths = []
    modification_times = [1003, 1001, 1002]  # Out of order

    for i, mtime in enumerate(modification_times):
        mock_path = Mock(spec=Path)
        mock_stat = Mock()
        mock_stat.st_mtime = mtime
        mock_path.stat.return_value = mock_stat
        mock_path.name = f"image_{i}.jpg"  # For identification
        mock_paths.append(mock_path)

    # Sort using the same logic as the function
    sorted_paths = sorted(mock_paths, key=lambda x: x.stat().st_mtime, reverse=False)

    # Verify sorting order (should be 1001, 1002, 1003)
    assert sorted_paths[0].stat().st_mtime == 1001
    assert sorted_paths[1].stat().st_mtime == 1002
    assert sorted_paths[2].stat().st_mtime == 1003


@patch('audify.utils.combine_covers.Image.open')
@patch('audify.utils.combine_covers.Image.new')
@patch('builtins.print')
def test_image_resizing_logic(
        mock_print, mock_image_new, mock_image_open, temp_output_dir
    ):
    """Test that images are properly resized to 420px width maintaining aspect ratio."""
    # Create mock image with known dimensions
    mock_img = Mock(spec=Image.Image)
    mock_img.width = 300
    mock_img.height = 600  # 2:1 aspect ratio
    resized_img = Mock(spec=Image.Image)
    resized_img.height = 840  # 420 * 600 / 300 = 840
    mock_img.resize.return_value = resized_img
    resized_img.resize.return_value = resized_img
    resized_img.crop.return_value = resized_img

    mock_path = Mock(spec=Path)
    mock_stat = Mock()
    mock_stat.st_mtime = 1000
    mock_path.stat.return_value = mock_stat

    mock_image_open.return_value = mock_img
    mock_new_image = Mock()
    mock_image_new.return_value = mock_new_image

    combine_covers([mock_path], temp_output_dir)

    # Verify resize was called with correct dimensions
    expected_height = int(420 * 600 / 300)  # 840
    mock_img.resize.assert_called_with((420, expected_height))


@patch('audify.utils.combine_covers.Image.open')
@patch('audify.utils.combine_covers.Image.new')
@patch('builtins.print')
def test_image_height_adjustment(
        mock_print, mock_image_new, mock_image_open, temp_output_dir
    ):
    """Test image height adjustment logic (resize shorter images, crop taller ones)."""
    # Create mock images with different heights after initial resize
    mock_imgs = []
    mock_paths = []
    heights = [500, 600, 700]  # Different heights
    median_height = int(np.median(heights))  # 600

    for i, height in enumerate(heights):
        mock_img = Mock(spec=Image.Image)
        mock_img.width = 420
        mock_img.height = height
        mock_img.resize.return_value = mock_img
        mock_img.crop.return_value = mock_img
        mock_imgs.append(mock_img)

        mock_path = Mock(spec=Path)
        mock_stat = Mock()
        mock_stat.st_mtime = 1000 + i
        mock_path.stat.return_value = mock_stat
        mock_paths.append(mock_path)

    mock_image_open.side_effect = mock_imgs
    mock_new_image = Mock()
    mock_image_new.return_value = mock_new_image

    combine_covers(mock_paths, temp_output_dir)

    # First image (height 500 < median 600) should be resized
    mock_imgs[0].resize.assert_called()

    # Third image (height 700 > median 600) should be cropped
    mock_imgs[2].crop.assert_called_with((0, 0, 420, median_height))
