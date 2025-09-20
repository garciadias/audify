"""
Shared audio utilities for common audio processing operations.

This module consolidates audio operations that are repeated across different
modules to reduce code duplication and provide consistent audio handling.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import tqdm
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Utility class for common audio processing operations."""

    @staticmethod
    def get_duration(file_path: str | Path) -> float:
        """
        Get the duration of an audio file in seconds.

        Args:
            file_path: Path to the audio file

        Returns:
            Duration in seconds, or 0.0 if the file cannot be decoded
        """
        try:
            audio = AudioSegment.from_file(str(file_path))
            return len(audio) / 1000.0
        except CouldntDecodeError:
            logger.error(f"Could not decode audio file: {file_path}")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting duration for {file_path}: {e}")
            return 0.0

    @staticmethod
    def convert_wav_to_mp3(
        wav_path: Path,
        bitrate: str = "192k",
        remove_original: bool = True,
    ) -> Path:
        """
        Convert a WAV file to MP3 format.

        Args:
            wav_path: Path to the WAV file
            bitrate: MP3 bitrate (default: "192k")
            remove_original: Whether to remove the original WAV file

        Returns:
            Path to the created MP3 file

        Raises:
            FileNotFoundError: If the WAV file doesn't exist
            Exception: If conversion fails
        """
        mp3_path = wav_path.with_suffix(".mp3")
        logger.info(f"Converting {wav_path.name} to MP3...")

        try:
            audio = AudioSegment.from_wav(str(wav_path))
            audio.export(str(mp3_path), format="mp3", bitrate=bitrate)
            logger.info(f"MP3 conversion successful: {mp3_path.name}")

            if remove_original:
                wav_path.unlink(missing_ok=True)

            return mp3_path
        except FileNotFoundError:
            logger.error(f"WAV file not found for conversion: {wav_path}")
            raise
        except Exception as e:
            logger.error(f"Error converting {wav_path.name} to MP3: {e}", exc_info=True)
            raise

    @staticmethod
    def combine_audio_files(
        file_paths: List[Path],
        output_path: Path,
        output_format: str = "wav",
        show_progress: bool = True,
        description: str = "Combining Audio",
    ) -> AudioSegment:
        """
        Combine multiple audio files into a single audio file.

        Args:
            file_paths: List of paths to audio files to combine
            output_path: Path for the output combined file
            output_format: Output format ("wav", "mp3", "mp4")
            show_progress: Whether to show progress bar
            description: Description for progress bar

        Returns:
            Combined AudioSegment

        Raises:
            ValueError: If no valid audio files are found
        """
        combined_audio = AudioSegment.empty()
        valid_files = 0

        file_iterator = (
            tqdm.tqdm(file_paths, desc=description, unit="file")
            if show_progress
            else file_paths
        )

        for file_path in file_iterator:
            try:
                if file_path.suffix.lower() == ".mp3":
                    audio = AudioSegment.from_mp3(str(file_path))
                elif file_path.suffix.lower() == ".wav":
                    audio = AudioSegment.from_wav(str(file_path))
                else:
                    audio = AudioSegment.from_file(str(file_path))

                combined_audio += audio
                valid_files += 1
            except CouldntDecodeError:
                logger.error(f"Could not decode audio file: {file_path}, skipping.")
            except FileNotFoundError:
                logger.warning(f"Audio file not found: {file_path}, skipping.")
            except Exception as e:
                logger.error(
                    f"Error processing audio file {file_path}: {e}, skipping.",
                    exc_info=True,
                )

        if valid_files == 0:
            raise ValueError("No valid audio files found to combine")

        if len(combined_audio) == 0:
            raise ValueError("Combined audio is empty")

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine export parameters based on format
        export_params = {"format": output_format}
        if output_format == "mp3":
            export_params["bitrate"] = "192k"
        elif output_format == "mp4":
            export_params["codec"] = "aac"
            export_params["bitrate"] = "64k"

        logger.info(f"Exporting combined audio to {output_path}")
        combined_audio.export(str(output_path), **export_params)

        return combined_audio

    @staticmethod
    def split_audio_by_duration(
        file_paths: List[Path], max_duration_hours: float = 15.0
    ) -> List[List[Path]]:
        """
        Split a list of audio files into chunks based on total duration.

        Args:
            file_paths: List of audio file paths
            max_duration_hours: Maximum duration per chunk in hours

        Returns:
            List of chunks, where each chunk is a list of file paths
        """
        max_duration_seconds = max_duration_hours * 3600
        chunks = []
        current_chunk: List[Path] = []
        current_duration = 0.0

        for file_path in file_paths:
            try:
                file_duration = AudioProcessor.get_duration(file_path)

                # If adding this file would exceed the limit, start a new chunk
                if (
                    current_chunk
                    and (current_duration + file_duration) > max_duration_seconds
                ):
                    chunks.append(current_chunk)
                    current_chunk = [file_path]
                    current_duration = file_duration
                else:
                    current_chunk.append(file_path)
                    current_duration += file_duration

            except Exception as e:
                logger.warning(
                    f"Could not get duration for {file_path}: {e}, "
                    "adding to current chunk anyway"
                )
                current_chunk.append(file_path)

        # Add the last chunk if it has files
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    @staticmethod
    def create_temp_audio_file(
        file_paths: List[Path],
        output_prefix: str,
        output_format: str = "mp4",
        temp_dir: Optional[Path] = None,
    ) -> Path:
        """
        Create a temporary audio file from multiple source files.

        Args:
            file_paths: List of audio file paths to combine
            output_prefix: Prefix for the temporary file name
            output_format: Format for the temporary file
            temp_dir: Directory for temporary file (uses system temp if None)

        Returns:
            Path to the created temporary file
        """
        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir())

        temp_path = temp_dir / f"{output_prefix}.tmp.{output_format}"

        if temp_path.exists():
            logger.info(f"Temporary audio file already exists: {temp_path}")
            return temp_path

        logger.info(f"Creating temporary audio file: {temp_path}")
        AudioProcessor.combine_audio_files(
            file_paths=file_paths,
            output_path=temp_path,
            output_format=output_format,
            description=f"Creating {output_prefix}",
        )

        return temp_path
