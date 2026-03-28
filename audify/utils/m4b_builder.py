"""
Shared M4B audiobook assembly utilities.

This module consolidates the FFmpeg-based M4B creation logic that was previously
duplicated between ``EpubSynthesizer`` and ``AudiobookCreator``.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import IO, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard FFmpeg metadata header written to every chapter file.
_FFMETADATA_HEADER = (
    ";FFMETADATA1\n"
    "major_brand=M4A\n"
    "minor_version=512\n"
    "compatible_brands=M4A isis2\n"
    "encoder=Lavf61.7.100\n"
)


def write_metadata_header(metadata_path: Path) -> None:
    """Write the standard FFmpeg metadata header to *metadata_path*."""
    logger.info(f"Initializing metadata file: {metadata_path}")
    try:
        with open(metadata_path, "w") as f:
            f.write(_FFMETADATA_HEADER)
    except IOError as e:
        logger.error(f"Failed to initialize metadata file: {e}", exc_info=True)
        raise


def append_chapter_metadata(
    metadata_path: Path,
    title: str,
    start_ms: int,
    duration_s: float,
) -> int:
    """Append a single chapter entry to *metadata_path*.

    Args:
        metadata_path: Path to the FFmpeg metadata file.
        title: Chapter title.
        start_ms: Chapter start time in milliseconds.
        duration_s: Chapter duration in seconds.

    Returns:
        End time in milliseconds (= next chapter's start time).
    """
    end_ms = start_ms + int(duration_s * 1000)
    cleaned_title = title.replace("\n", " ").replace("\r", "")
    logger.debug(
        f"Appending chapter '{cleaned_title}': start={start_ms}, end={end_ms}"
    )
    try:
        with open(metadata_path, "a") as f:
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_ms}\n")
            f.write(f"END={end_ms}\n")
            f.write(f"title={cleaned_title}\n")
        return end_ms
    except IOError as e:
        logger.error(
            f"Failed to write chapter metadata for '{cleaned_title}': {e}",
            exc_info=True,
        )
        raise


def build_ffmpeg_command(
    input_m4b: Path,
    metadata_path: Path,
    output_m4b: Path,
    cover_image: Optional[Path] = None,
) -> Tuple[List[str], Optional[IO[bytes]]]:
    """Build the FFmpeg command to assemble the final M4B file.

    A temporary copy of the cover image is created (when provided) because
    FFmpeg may lock the source file on some systems.

    Args:
        input_m4b: Temporary M4B with raw audio.
        metadata_path: FFmpeg metadata file with chapter markers.
        output_m4b: Destination M4B path.
        cover_image: Optional cover image path.

    Returns:
        ``(command_list, cover_temp_file)`` — caller is responsible for closing
        and deleting *cover_temp_file* after ``subprocess.run``.
    """
    cover_args: List[str] = []
    cover_temp_file: Optional[tempfile._TemporaryFileWrapper] = None

    if cover_image and isinstance(cover_image, Path) and cover_image.exists():
        cover_temp_file = tempfile.NamedTemporaryFile(
            suffix=cover_image.suffix, delete=False
        )
        shutil.copy(cover_image, cover_temp_file.name)
        logger.info(f"Using cover image: {cover_image}")
        cover_args = [
            "-i", cover_temp_file.name,
            "-map", "0:a",
            "-map", "2:v",
            "-disposition:v", "attached_pic",
            "-c:v", "copy",
        ]
    else:
        if cover_image:
            logger.warning(f"Cover image not found: {cover_image}")
        cover_args = ["-map", "0:a"]

    command = [
        "ffmpeg",
        "-i", str(input_m4b),
        "-i", str(metadata_path),
        *cover_args,
        "-map_metadata", "1",
        "-c:a", "copy",
        "-f", "mp4",
        "-y",
        str(output_m4b),
    ]
    return command, cover_temp_file


def run_ffmpeg(command: List[str]) -> None:
    """Execute *command* via ``subprocess.run``.

    Raises:
        subprocess.CalledProcessError: If FFmpeg exits with a non-zero code.
        FileNotFoundError: If FFmpeg is not installed / not in PATH.
    """
    logger.debug(f"Running FFmpeg: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("FFmpeg completed successfully.")
        logger.debug(f"FFmpeg stdout:\n{result.stdout}")
        logger.debug(f"FFmpeg stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed (exit {e.returncode})")
        logger.error(f"FFmpeg stdout:\n{e.stdout}")
        logger.error(f"FFmpeg stderr:\n{e.stderr}")
        raise
    except FileNotFoundError:
        logger.error(
            "FFmpeg not found. Please ensure FFmpeg is installed and in PATH."
        )
        raise


def assemble_m4b(
    input_m4b: Path,
    metadata_path: Path,
    output_m4b: Path,
    cover_image: Optional[Path] = None,
) -> None:
    """High-level helper: build command, run FFmpeg, clean up temp cover.

    After a successful run the *input_m4b* temp file is deleted.

    Args:
        input_m4b: Temporary M4B with raw audio (deleted on success).
        metadata_path: FFmpeg metadata file with chapter markers.
        output_m4b: Destination M4B path.
        cover_image: Optional cover image path.
    """
    command, cover_temp_file = build_ffmpeg_command(
        input_m4b, metadata_path, output_m4b, cover_image
    )
    try:
        run_ffmpeg(command)
        input_m4b.unlink(missing_ok=True)
        logger.info(f"M4B created: {output_m4b}")
    finally:
        if cover_temp_file:
            try:
                cover_temp_file.close()
                Path(cover_temp_file.name).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Error cleaning up temp cover: {e}")
