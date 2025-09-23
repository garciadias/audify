import contextlib
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import tqdm
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from audify.readers.ebook import EpubReader
from audify.readers.pdf import PdfReader
from audify.translate import translate_sentence
from audify.utils.api_config import KokoroAPIConfig
from audify.utils.audio import AudioProcessor
from audify.utils.constants import (
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    KOKORO_API_BASE_URL,
    LANG_CODES,
    OUTPUT_BASE_DIR,
)
from audify.utils.logging_utils import setup_logging
from audify.utils.text import break_text_into_sentences, get_file_name_title

# Configure logging
logger = setup_logging(module_name=__name__)

MODULE_PATH = Path(__file__).resolve().parents[1]

# Mute specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@contextlib.contextmanager
def suppress_stdout():
    """Temporarily suppress stdout."""
    original_stdout = sys.stdout
    log_path = Path(tempfile.gettempdir()) / "audify_stdout_suppress.log"
    with open(log_path, "w") as f_null:
        sys.stdout = f_null
        try:
            yield
        finally:
            sys.stdout = original_stdout


class BaseSynthesizer:
    """Base class for text-to-speech synthesis."""

    def __init__(
        self,
        path: str | Path,
        voice: str,
        translate: Optional[str],
        save_text: bool,
        language: str = "en",
        model_name: str = DEFAULT_MODEL,
    ):
        self.path = Path(path).resolve()
        self.language = language
        self.speaker = voice
        self.translate = translate
        self.model_name = model_name
        self.save_text = save_text
        self.tmp_dir_context = tempfile.TemporaryDirectory(
            prefix=f"audify_{self.path.stem}_"
        )
        self.tmp_dir = Path(self.tmp_dir_context.name)

    def _synthesize_kokoro(self, sentences: List[str], output_wav_path: Path) -> None:
        """Synthesize sentences using the Kokoro API."""
        # Initialize API config
        api_config = KokoroAPIConfig()
        combined_audio = AudioSegment.empty()
        temp_audio_files: List[Path] = []

        try:
            logger.info("Starting Kokoro API synthesis...")

            # Check if API is available
            try:
                response = requests.get(api_config.voices_url, timeout=5)
                if response.status_code != 200:
                    raise requests.RequestException(
                        f"API returned status {response.status_code}"
                    )
                available_voices = response.json().get("voices", [])
                logger.info(
                    f"Connected to Kokoro API. Available voices: "
                    f"{len(available_voices)}"
                )
            except requests.RequestException as e:
                logger.error(
                    f"Failed to connect to Kokoro API at {api_config.base_url}: {e}"
                )
                raise

            for i, sentence in tqdm.tqdm(
                enumerate(sentences),
                desc="Kokoro Synthesizing",
                total=len(sentences),
                unit="sentence",
            ):
                if not sentence.strip():
                    continue

                try:
                    # Make API request for each sentence
                    if self.speaker not in available_voices:
                        raise ValueError(
                            f"Speaker '{self.speaker}' not available in Kokoro voices."
                        )
                    if self.language not in LANG_CODES.keys():
                        raise ValueError(
                            f"Language code '{self.language}' is not supported."
                        )
                    response = requests.post(
                        api_config.speech_url,
                        json={
                            "model": "kokoro",
                            "input": sentence,
                            "voice": self.speaker,
                            "response_format": "wav",
                            "lang_code": LANG_CODES[self.translate or self.language],
                            "speed": 1.0,
                        },
                        timeout=api_config.timeout,
                    )

                    if response.status_code == 200:
                        temp_wav_path = self.tmp_dir / f"segment_{i}.wav"
                        with open(temp_wav_path, "wb") as f:
                            f.write(response.content)
                        temp_audio_files.append(temp_wav_path)
                    else:
                        logger.warning(
                            f"API request failed for sentence {i}: "
                            f"{response.status_code}"
                        )
                        continue

                except requests.RequestException as e:
                    logger.warning(f"Request failed for sentence {i}: {e}")
                    continue

            logger.info("Combining Kokoro audio segments...")
            for temp_wav_path in tqdm.tqdm(
                temp_audio_files, desc="Combining Segments", unit="file"
            ):
                if temp_wav_path.exists():
                    try:
                        segment = AudioSegment.from_wav(temp_wav_path)
                        combined_audio += segment
                    except CouldntDecodeError:
                        logger.warning(
                            f"Could not decode temporary segment: {temp_wav_path}"
                        )
                    finally:
                        temp_wav_path.unlink(missing_ok=True)
                else:
                    logger.warning(f"Temporary segment file not found: {temp_wav_path}")

            logger.info(f"Exporting combined Kokoro audio to {output_wav_path}")
            combined_audio.export(output_wav_path, format="wav")

        except Exception as e:
            logger.error(f"Error during Kokoro API synthesis: {e}", exc_info=True)
            for temp_file in temp_audio_files:
                temp_file.unlink(missing_ok=True)
            raise

    def _synthesize_sentences(
        self, sentences: List[str], output_wav_path: Path
    ) -> None:
        """Synthesize a list of sentences into a single WAV file."""
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        self._synthesize_kokoro(sentences, output_wav_path)

        logger.info(f"Raw WAV synthesis complete: {output_wav_path}")

    def _convert_to_mp3(self, wav_path: Path) -> Path:
        """Converts a WAV file to MP3 and removes the original WAV."""
        return AudioProcessor.convert_wav_to_mp3(wav_path)

    def synthesize(self) -> Path:
        """Abstract method for the main synthesis process."""
        raise NotImplementedError("Subclasses must implement the synthesize method.")

    def __del__(self):
        """Ensure temporary directory is cleaned up."""
        if hasattr(self, "tmp_dir_context"):
            try:
                self.tmp_dir_context.cleanup()
                logger.debug(f"Cleaned up temporary directory: {self.tmp_dir}")
            except Exception as e:
                logger.error(
                    f"Error cleaning up temporary directory {self.tmp_dir}: {e}"
                )

    def stop(self) -> None:
        """Stops the synthesis process if running."""
        logger.info("Stopping synthesis process...")
        # Note: API-based synthesis doesn't have a persistent process to stop
        # This method is kept for interface compatibility
        logger.info("Synthesis process stopped (API-based).")

    def get_terminal_output(self) -> str:
        """Returns the terminal output of the synthesis process."""
        log_path = Path(tempfile.gettempdir()) / "audify_stdout_suppress.log"
        if log_path.exists():
            with open(log_path, "r") as f:
                return f.read()
        else:
            return "No terminal output available."


class EpubSynthesizer(BaseSynthesizer):
    """Synthesizer for EPUB files, creating an M4B audiobook."""

    def __init__(
        self,
        path: str | Path,
        language: Optional[str] = None,
        speaker: str = DEFAULT_SPEAKER,
        translate: Optional[str] = None,
        save_text: bool = False,
        confirm: bool = True,
        model_name: str = DEFAULT_MODEL,
    ):
        self.reader = EpubReader(path)
        detected_language = self.reader.get_language()
        resolved_language = language or detected_language
        self.output_base_dir = Path(OUTPUT_BASE_DIR).resolve()
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
        if not resolved_language:
            raise ValueError(
                "Language must be provided or detectable from the EPUB metadata."
            )

        self.title = self.reader.title
        if translate:
            logger.info(f"Translating title from {resolved_language} to {translate}")
            self.title = translate_sentence(
                sentence=self.title, src_lang=resolved_language, tgt_lang=translate
            )

        self.file_name = get_file_name_title(self.title)
        self._setup_paths(self.file_name)
        self._initialize_metadata_file()
        self.confirm = confirm

        super().__init__(
            path=path,
            language=resolved_language,
            voice=speaker,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
        )

        self.cover_image_path: Optional[Path] = self.reader.get_cover_image(
            self.audiobook_path
        )

    def _setup_paths(self, file_name_base: str) -> None:
        """Sets up the necessary output paths."""
        if not self.output_base_dir.exists():
            logger.info(f"Creating output base directory: {self.output_base_dir}")
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.audiobook_path = self.output_base_dir / file_name_base
        self.audiobook_path.mkdir(parents=True, exist_ok=True)
        self.list_of_contents_path = self.audiobook_path / "chapters.txt"
        self.final_m4b_path = self.audiobook_path / f"{file_name_base}.m4b"
        self.temp_m4b_path = self.audiobook_path / f"{file_name_base}.tmp.m4b"
        logger.info(f"Output directory set to: {self.audiobook_path}")

    def _initialize_metadata_file(self) -> None:
        """Writes the initial header to the FFmpeg metadata file."""
        logger.info(f"Initializing metadata file: {self.list_of_contents_path}")
        try:
            with open(self.list_of_contents_path, "w") as f:
                f.write(";FFMETADATA1\n")
                f.write("major_brand=M4A\n")
                f.write("minor_version=512\n")
                f.write("compatible_brands=M4A isis2\n")
                f.write("encoder=Lavf61.7.100\n")
        except IOError as e:
            logger.error(f"Failed to initialize metadata file: {e}", exc_info=True)
            raise

    def synthesize_chapter(self, chapter_content: str, chapter_number: int) -> Path:
        """Synthesizes a single chapter into an MP3 file."""
        logger.info(f"Synthesizing Chapter {chapter_number:03d}...")
        chapter_wav_path = self.audiobook_path / f"chapter_{chapter_number:03d}.wav"
        chapter_mp3_path = chapter_wav_path.with_suffix(".mp3")

        if chapter_mp3_path.exists():
            logger.info(
                f"Chapter {chapter_number:03d} MP3 already exists, skipping synthesis."
            )
            return chapter_mp3_path

        chapter_txt = self.reader.extract_text(chapter_content)
        sentences = break_text_into_sentences(chapter_txt)

        if not sentences:
            logger.warning(
                f"Chapter {chapter_number:03d} contains no text to synthesize."
            )
            return chapter_mp3_path

        if self.translate and self.language:
            logger.info(
                f"Translating {len(sentences)} sentences for Chapter"
                f" {chapter_number:03d}..."
            )
            try:
                sentences = [
                    translate_sentence(
                        sentence, src_lang=self.language, tgt_lang=self.translate
                    )
                    for sentence in tqdm.tqdm(
                        sentences,
                        desc=f"Translating Ch. {chapter_number:03d}",
                        unit="sentence",
                    )
                ]
            except Exception as e:
                logger.error(
                    f"Error translating chapter {chapter_number:03d}: {e}",
                    exc_info=True,
                )
                logger.warning(
                    "Proceeding with original text for synthesis due to translation"
                    " error."
                )
                sentences = break_text_into_sentences(chapter_txt)

        self._synthesize_sentences(sentences, chapter_wav_path)
        return self._convert_to_mp3(chapter_wav_path)

    def _calculate_total_duration(self, mp3_files: List[Path]) -> float:
        """Calculate total duration of MP3 files in seconds."""
        total_duration = 0.0
        for mp3_file in mp3_files:
            duration = AudioProcessor.get_duration(str(mp3_file))
            total_duration += duration
        return total_duration

    def _split_chapters_by_duration(
        self, chapter_mp3_files: List[Path], max_hours: float = 15.0
    ) -> List[List[Path]]:
        """Split chapter MP3 files into chunks with maximum duration in hours."""
        return AudioProcessor.split_audio_by_duration(chapter_mp3_files, max_hours)

    def _create_temp_m4b_for_chunk(
        self, chunk_files: List[Path], chunk_index: int
    ) -> Path:
        """Create a temporary M4B file for a specific chunk of chapters."""
        chunk_temp_path = (
            self.audiobook_path / f"{self.file_name}_part{chunk_index + 1}.tmp.m4b"
        )

        if chunk_temp_path.exists():
            logger.info(
                f"Temporary M4B for chunk {chunk_index + 1} already exists: "
                f"{chunk_temp_path}"
            )
            return chunk_temp_path

        logger.info(
            f"Combining {len(chunk_files)} chapter MP3s for chunk {chunk_index + 1}..."
        )
        combined_audio = AudioSegment.empty()

        for mp3_file in tqdm.tqdm(
            chunk_files, desc=f"Combining Chunk {chunk_index + 1}", unit="file"
        ):
            try:
                audio = AudioSegment.from_mp3(mp3_file)
                combined_audio += audio
            except CouldntDecodeError:
                logger.error(f"Could not decode chapter file: {mp3_file}, skipping.")
            except Exception as e:
                logger.error(
                    f"Error processing chapter file {mp3_file}: {e}, skipping.",
                    exc_info=True,
                )

        if len(combined_audio) == 0:
            logger.error(f"Combined audio for chunk {chunk_index + 1} is empty.")
            return chunk_temp_path

        logger.info(f"Exporting combined audio to temporary M4B: {chunk_temp_path}")
        try:
            combined_audio.export(
                chunk_temp_path, format="mp4", codec="aac", bitrate="64k"
            )
        except Exception as e:
            logger.error(
                f"Failed to export temporary M4B file for chunk {chunk_index + 1}: {e}",
                exc_info=True,
            )
            chunk_temp_path.unlink(missing_ok=True)
            raise

        return chunk_temp_path

    def _create_metadata_for_chunk(
        self, chunk_files: List[Path], chunk_index: int
    ) -> Path:
        """Create a metadata file for a specific chunk."""
        chunk_metadata_path = (
            self.audiobook_path / f"chapters_part{chunk_index + 1}.txt"
        )

        logger.info(
            f"Creating metadata file for chunk {chunk_index + 1}: {chunk_metadata_path}"
        )
        try:
            with open(chunk_metadata_path, "w") as f:
                f.write(";FFMETADATA1\n")
                f.write("major_brand=M4A\n")
                f.write("minor_version=512\n")
                f.write("compatible_brands=M4A isis2\n")
                f.write("encoder=Lavf61.7.100\n")

                current_start_time_ms = 0
                for mp3_file in chunk_files:
                    try:
                        # Extract chapter number from filename
                        chapter_num = int(mp3_file.stem.split("_")[1])
                        duration = AudioProcessor.get_duration(str(mp3_file))

                        if duration > 0:
                            end_time_ms = current_start_time_ms + int(duration * 1000)
                            f.write("[CHAPTER]\n")
                            f.write("TIMEBASE=1/1000\n")
                            f.write(f"START={current_start_time_ms}\n")
                            f.write(f"END={end_time_ms}\n")
                            f.write(f"title=Chapter {chapter_num}\n")
                            current_start_time_ms = end_time_ms
                    except Exception as e:
                        logger.warning(
                            f"Could not process metadata for {mp3_file}: {e}"
                        )

        except IOError as e:
            logger.error(
                f"Failed to create metadata file for chunk {chunk_index + 1}: {e}",
                exc_info=True,
            )
            raise

        return chunk_metadata_path

    def create_m4b(self) -> None:
        """Combines chapter MP3s and metadata into M4B file(s),
        splitting if necessary."""
        logger.info("Starting M4B creation process...")
        chapter_mp3_files = sorted(self.audiobook_path.glob("chapter_*.mp3"))

        if not chapter_mp3_files:
            logger.error("No chapter MP3 files found to create M4B.")
            return

        # Calculate total duration and check if we need to split
        total_duration_hours = self._calculate_total_duration(chapter_mp3_files) / 3600
        logger.info(f"Total audiobook duration: {total_duration_hours:.2f} hours")

        # If duration is less than 15 hours, create a single M4B
        if total_duration_hours <= 15.0:
            logger.info("Creating single M4B file (duration <= 15 hours)")
            self._create_single_m4b(chapter_mp3_files)
        else:
            logger.info(
                f"Duration ({total_duration_hours:.2f}h) exceeds 15 hours, "
                f"splitting into multiple M4B files"
            )
            self._create_multiple_m4bs(chapter_mp3_files)

    def _create_single_m4b(self, chapter_mp3_files: List[Path]) -> None:
        """Create a single M4B file from all chapters."""
        if not self.temp_m4b_path.exists():
            logger.info(f"Combining {len(chapter_mp3_files)} chapter MP3s...")
            combined_audio = AudioSegment.empty()
            for mp3_file in tqdm.tqdm(
                chapter_mp3_files, desc="Combining Chapters", unit="file"
            ):
                try:
                    audio = AudioSegment.from_mp3(mp3_file)
                    combined_audio += audio
                except CouldntDecodeError:
                    logger.error(
                        f"Could not decode chapter file: {mp3_file}, skipping."
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing chapter file {mp3_file}: {e}, skipping.",
                        exc_info=True,
                    )

            if len(combined_audio) == 0:
                logger.error("Combined audio is empty. Cannot create M4B.")
                return

            logger.info(
                f"Exporting combined audio to temporary M4B: {self.temp_m4b_path}"
            )
            try:
                combined_audio.export(
                    self.temp_m4b_path, format="mp4", codec="aac", bitrate="64k"
                )
            except Exception as e:
                logger.error(f"Failed to export temporary M4B file: {e}", exc_info=True)
                self.temp_m4b_path.unlink(missing_ok=True)
                raise
        else:
            logger.info(
                f"Temporary M4B file already exists: {self.temp_m4b_path}."
                " Skipping combination."
            )

        logger.info("Adding metadata and cover image using FFmpeg...")
        ffmpeg_command, cover_temp_file = self._build_ffmpeg_command(chapter_mp3_files)

        try:
            logger.debug(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
            result = subprocess.run(
                ffmpeg_command, check=True, capture_output=True, text=True
            )
            logger.info("FFmpeg process completed successfully.")
            logger.debug(f"FFmpeg stdout:\n{result.stdout}")
            logger.debug(f"FFmpeg stderr:\n{result.stderr}")

            logger.info(f"Cleaning up temporary file: {self.temp_m4b_path}")
            self.temp_m4b_path.unlink(missing_ok=True)

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed with exit code {e.returncode}")
            logger.error(f"FFmpeg stdout:\n{e.stdout}")
            logger.error(f"FFmpeg stderr:\n{e.stderr}")
            logger.error(
                "M4B creation failed. The temporary M4B file and chapter files are"
                " preserved for inspection."
            )
            raise
        except FileNotFoundError:
            logger.error(
                "FFmpeg command not found. Please ensure FFmpeg is installed and in"
                " your system's PATH."
            )
            raise
        finally:
            if cover_temp_file:
                try:
                    cover_temp_file.close()
                    Path(cover_temp_file.name).unlink(missing_ok=True)
                    logger.debug(
                        f"Cleaned up temporary cover file: {cover_temp_file.name}"
                    )
                except Exception as e_clean:
                    logger.warning(
                        f"Error cleaning up temporary cover file"
                        f" {cover_temp_file.name}: {e_clean}"
                    )

    def _create_multiple_m4bs(self, chapter_mp3_files: List[Path]) -> None:
        """Create multiple M4B files by splitting chapters into chunks."""
        chunks = self._split_chapters_by_duration(chapter_mp3_files, max_hours=15.0)
        logger.info(f"Split into {len(chunks)} chunks")

        for chunk_index, chunk_files in enumerate(chunks):
            chunk_duration = self._calculate_total_duration(chunk_files) / 3600
            logger.info(
                f"Processing chunk {chunk_index + 1}/{len(chunks)} "
                f"({len(chunk_files)} chapters, {chunk_duration:.2f}h)"
            )

            # Create temporary M4B for this chunk
            chunk_temp_path = self._create_temp_m4b_for_chunk(chunk_files, chunk_index)

            if not chunk_temp_path.exists():
                logger.error(
                    f"Failed to create temporary M4B for chunk {chunk_index + 1}"
                )
                continue

            # Create metadata for this chunk
            chunk_metadata_path = self._create_metadata_for_chunk(
                chunk_files, chunk_index
            )

            # Create final M4B file for this chunk
            chunk_final_path = (
                self.audiobook_path / f"{self.file_name}_part{chunk_index + 1}.m4b"
            )

            logger.info(
                f"Creating final M4B for chunk {chunk_index + 1}: {chunk_final_path}"
            )
            ffmpeg_command, cover_temp_file = self._build_ffmpeg_command_for_chunk(
                chunk_temp_path, chunk_metadata_path, chunk_final_path
            )

            try:
                logger.debug(
                    f"Running FFmpeg command for chunk {chunk_index + 1}: "
                    f"{' '.join(ffmpeg_command)}"
                )
                result = subprocess.run(
                    ffmpeg_command, check=True, capture_output=True, text=True
                )
                logger.info(
                    f"FFmpeg process completed successfully for chunk "
                    f"{chunk_index + 1}."
                )
                logger.debug(f"FFmpeg stdout:\n{result.stdout}")
                logger.debug(f"FFmpeg stderr:\n{result.stderr}")

                # Clean up temporary files for this chunk
                chunk_temp_path.unlink(missing_ok=True)
                chunk_metadata_path.unlink(missing_ok=True)

            except subprocess.CalledProcessError as e:
                logger.error(
                    f"FFmpeg command failed for chunk {chunk_index + 1} "
                    f"with exit code {e.returncode}"
                )
                logger.error(f"FFmpeg stdout:\n{e.stdout}")
                logger.error(f"FFmpeg stderr:\n{e.stderr}")
                logger.error(f"M4B creation failed for chunk {chunk_index + 1}.")
            except FileNotFoundError:
                logger.error(
                    "FFmpeg command not found. Please ensure FFmpeg is installed and in"
                    " your system's PATH."
                )
            finally:
                if cover_temp_file:
                    try:
                        cover_temp_file.close()
                        Path(cover_temp_file.name).unlink(missing_ok=True)
                    except Exception as e_clean:
                        logger.warning(
                            f"Error cleaning up temporary cover file: {e_clean}"
                        )

        logger.info(f"Created {len(chunks)} M4B files for long audiobook")

    def _build_ffmpeg_command(
        self, chapter_files: List[Path]
    ) -> Tuple[List[str], Optional[tempfile._TemporaryFileWrapper]]:
        """Builds the FFmpeg command for M4B creation."""
        cover_args = []
        cover_temp_file = None
        if isinstance(self.cover_image_path, Path) and self.cover_image_path.exists():
            cover_temp_file = tempfile.NamedTemporaryFile(
                suffix=self.cover_image_path.suffix, delete=False
            )
            shutil.copy(self.cover_image_path, cover_temp_file.name)
            logger.info(f"Using cover image: {self.cover_image_path}")
            cover_args = [
                "-i",
                cover_temp_file.name,
                "-map",
                "0:a",
                "-map",
                "2:v",
                "-disposition:v",
                "attached_pic",
                "-c:v",
                "copy",
            ]
        else:
            logger.warning("No cover image found or provided.")
            cover_args = [
                "-map",
                "0:a",
            ]

        command = [
            "ffmpeg",
            "-i",
            str(self.temp_m4b_path),
            "-i",
            str(self.list_of_contents_path),
            *cover_args,
            "-map_metadata",
            "1",
            "-c:a",
            "copy",
            "-f",
            "mp4",
            "-y",
            str(self.final_m4b_path),
        ]

        return command, cover_temp_file

    def _build_ffmpeg_command_for_chunk(
        self, chunk_temp_path: Path, chunk_metadata_path: Path, chunk_final_path: Path
    ) -> Tuple[List[str], Optional[tempfile._TemporaryFileWrapper]]:
        """Builds the FFmpeg command for M4B creation of a specific chunk."""
        cover_args = []
        cover_temp_file = None
        if isinstance(self.cover_image_path, Path) and self.cover_image_path.exists():
            cover_temp_file = tempfile.NamedTemporaryFile(
                suffix=self.cover_image_path.suffix, delete=False
            )
            shutil.copy(self.cover_image_path, cover_temp_file.name)
            cover_args = [
                "-i",
                cover_temp_file.name,
                "-map",
                "0:a",
                "-map",
                "2:v",
                "-disposition:v",
                "attached_pic",
                "-c:v",
                "copy",
            ]
        else:
            cover_args = [
                "-map",
                "0:a",
            ]

        command = [
            "ffmpeg",
            "-i",
            str(chunk_temp_path),
            "-i",
            str(chunk_metadata_path),
            *cover_args,
            "-map_metadata",
            "1",
            "-c:a",
            "copy",
            "-f",
            "mp4",
            "-y",
            str(chunk_final_path),
        ]

        return command, cover_temp_file

    def _log_chapter_metadata(
        self, title: str, start_time_ms: int, duration_s: float
    ) -> int:
        """Appends chapter metadata to the FFmpeg metadata file."""
        end_time_ms = start_time_ms + int(duration_s * 1000)
        logger.debug(
            f"Logging metadata for '{title}': Start={start_time_ms}, End={end_time_ms},"
            f" Duration={duration_s:.2f}s"
        )
        try:
            with open(self.list_of_contents_path, "a") as f:
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={start_time_ms}\n")
                f.write(f"END={end_time_ms}\n")
                cleaned_title = title.replace("\n", " ").replace("\r", "")
                f.write(f"title={cleaned_title}\n")
            return end_time_ms
        except IOError as e:
            logger.error(
                f"Failed to write chapter metadata for '{title}': {e}", exc_info=True
            )
            return start_time_ms

    def _process_single_chapter(
        self, chapter_index: int, chapter_content: str, current_start_time_ms: int
    ) -> int:
        """Processes a single chapter: synthesizes (if needed) and logs metadata."""
        chapter_title_raw = self.reader.get_chapter_title(chapter_content)
        title = (
            f"Chapter {chapter_index}: {chapter_title_raw}"
            if chapter_title_raw
            else f"Chapter {chapter_index}"
        )

        chapter_mp3_path = self.audiobook_path / f"chapter_{chapter_index:03d}.mp3"
        duration_s = 0.0

        MIN_CHAPTER_LENGTH = 100
        is_too_short = len(chapter_content) < MIN_CHAPTER_LENGTH
        chapter_exists = chapter_mp3_path.exists()

        if is_too_short:
            logger.info(
                f"Skipping Chapter {chapter_index} ('{title}') - content too short."
            )
            return current_start_time_ms

        if chapter_exists:
            logger.info(f"Chapter {chapter_index} ('{title}') already synthesized.")
            try:
                duration_s = AudioProcessor.get_duration(str(chapter_mp3_path))
            except Exception as e:
                logger.warning(
                    f"Could not get duration for existing chapter {chapter_index}: {e}."
                    " Skipping metadata log for this chapter."
                )
                return current_start_time_ms
        else:
            logger.info(f"Processing Chapter {chapter_index} ('{title}').")
            try:
                synthesized_path = self.synthesize_chapter(
                    chapter_content, chapter_index
                )
                if synthesized_path.exists():
                    duration_s = AudioProcessor.get_duration(str(synthesized_path))
                else:
                    logger.warning(
                        f"Synthesized chapter {chapter_index} MP3 not found at"
                        f" {synthesized_path}. Cannot log metadata."
                    )
                    return current_start_time_ms
            except Exception as e:
                logger.error(
                    f"Failed to synthesize or get duration for chapter"
                    f" {chapter_index}: {e}",
                    exc_info=True,
                )
                return current_start_time_ms

        if duration_s > 0:
            next_start_time_ms = self._log_chapter_metadata(
                title, current_start_time_ms, duration_s
            )
            return next_start_time_ms
        else:
            logger.warning(
                f"Skipping metadata log for Chapter {chapter_index} due to zero or"
                " invalid duration."
            )
            return current_start_time_ms

    def process_chapters(self) -> None:
        """Iterates through EPUB chapters, synthesizes them, and logs metadata."""
        logger.info("Processing EPUB chapters...")
        chapters = self.reader.get_chapters()
        num_chapters = len(chapters)
        logger.info(f"Found {num_chapters} chapters.")

        current_chapter_start_time_ms = 0
        actual_chapter_id = 1

        for i, chapter_content in enumerate(
            tqdm.tqdm(chapters, desc="Processing Chapters", unit="chapter")
        ):
            if not chapter_content or (
                isinstance(chapter_content, str) and not chapter_content.strip()
            ):
                logger.warning(f"Skipping empty or invalid chapter data at index {i}.")
                continue

            MIN_CHAPTER_LENGTH = 100
            if len(chapter_content) >= MIN_CHAPTER_LENGTH:
                try:
                    current_chapter_start_time_ms = self._process_single_chapter(
                        actual_chapter_id,
                        chapter_content,
                        current_chapter_start_time_ms,
                    )
                    actual_chapter_id += 1
                except Exception as e:
                    logger.error(
                        f"Unhandled error processing chapter at index {i} (processed as"
                        f" chapter {actual_chapter_id}): {e}",
                        exc_info=True,
                    )

        logger.info(f"Finished processing {actual_chapter_id - 1} valid chapters.")

    def synthesize(self) -> Path:
        """Orchestrates the EPUB to M4B synthesis process."""
        logger.info(f"Starting synthesis for EPUB: {self.path.name}")
        self.process_chapters()
        self.create_m4b()
        logger.info(f"Audiobook synthesis complete: {self.final_m4b_path}")
        return self.final_m4b_path


class PdfSynthesizer(BaseSynthesizer):
    """Synthesizer for PDF files."""

    def __init__(
        self,
        pdf_path: str | Path,
        language: str = "en",
        model_name: str = DEFAULT_MODEL,
        speaker: str = DEFAULT_SPEAKER,
        output_dir: str | Path = OUTPUT_BASE_DIR,
        file_name: Optional[str] = None,
        translate: Optional[str] = None,
        save_text: bool = False,
    ):
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        output_base_name = file_name or pdf_path.stem
        self.output_wav_path = output_dir / f"{output_base_name}.wav"

        super().__init__(
            path=pdf_path,
            language=language,
            voice=speaker,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
        )

    def synthesize(self) -> Path:
        """Reads PDF, synthesizes text, and returns the path to the MP3."""
        logger.info(f"Starting synthesis for PDF: {self.path.name}")
        try:
            reader = PdfReader(self.path)
            logger.info("Extracting and cleaning text from PDF...")

            sentences = break_text_into_sentences(reader.cleaned_text)

            if not sentences:
                logger.warning("No text extracted from PDF. Cannot synthesize.")
                return self.output_wav_path.with_suffix(".mp3")

            logger.info(f"Extracted {len(sentences)} sentences.")

            if self.translate and self.language:
                logger.info(
                    f"Translating {len(sentences)} sentences to {self.translate}..."
                )
                try:
                    sentences = [
                        translate_sentence(
                            sentence, src_lang=self.language, tgt_lang=self.translate
                        )
                        for sentence in tqdm.tqdm(
                            sentences, desc="Translating PDF", unit="sentence"
                        )
                    ]
                except Exception as e:
                    logger.error(f"Error translating PDF content: {e}", exc_info=True)
                    logger.warning("Proceeding with original text for synthesis.")
                    sentences = break_text_into_sentences(reader.cleaned_text)

            self.output_wav_path.parent.mkdir(parents=True, exist_ok=True)

            self._synthesize_sentences(sentences, self.output_wav_path)
            final_mp3_path = self._convert_to_mp3(self.output_wav_path)
            logger.info(f"PDF synthesis complete: {final_mp3_path}")
            return final_mp3_path

        except Exception as e:
            logger.error(
                f"Error during PDF synthesis for {self.path.name}: {e}", exc_info=True
            )
            raise


class VoiceSamplesSynthesizer:
    """Synthesizer for creating voice samples with all available combinations."""

    def __init__(
        self,
        language: str = "en",
        translate: Optional[str] = None,
        sample_text: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.language = language
        self.translate = translate
        self.max_samples = max_samples
        self.sample_text = sample_text or (
            "Bean on bread is a simple yet delightful snack."
            "Hello, this is a sample of the text-to-speech synthesis. "
            "This sample demonstrates the quality and characteristics "
            "of this voice model combination. Each chapter in this audiobook "
            "represents a different voice and model pairing available."
        )

        # Set up paths
        self.output_path = OUTPUT_BASE_DIR / "voice_samples"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Set up temporary directory
        self.tmp_dir = Path(tempfile.mkdtemp())

        # Set up M4B paths
        self.temp_m4b_path = self.output_path / "voice_samples.tmp.m4b"
        self.final_m4b_path = self.output_path / "voice_samples.m4b"
        self.metadata_path = self.output_path / "voice_samples_metadata.txt"

    def _get_available_models_and_voices(self) -> Tuple[List[str], List[str]]:
        """Get available models and voices from Kokoro API."""
        try:
            # Get models
            models_response = requests.get(f"{KOKORO_API_BASE_URL}/models", timeout=10)
            models_response.raise_for_status()
            models_data = models_response.json().get("data", [])
            models = sorted([model.get("id") for model in models_data if "id" in model])

            # Get voices - use the API config to get the correct endpoint
            from audify.utils.api_config import KokoroAPIConfig
            api_config = KokoroAPIConfig()
            voices_response = requests.get(api_config.voices_url, timeout=10)
            voices_response.raise_for_status()
            voices_data = voices_response.json().get("voices", [])
            voices = sorted(voices_data)

            logger.info(f"Found {len(models)} models and {len(voices)} voices")
            return models, voices

        except requests.RequestException as e:
            logger.error(f"Error fetching models and voices from Kokoro API: {e}")
            return [], []

    def _create_sample_for_combination(
        self, model: str, voice: str, chapter_index: int
    ) -> Optional[Path]:
        """Create a sample audio file for a specific model-voice combination."""
        try:
            # Create a temporary synthesizer for this combination
            temp_synthesizer = BaseSynthesizer(
                path="sample.txt",
                voice=voice,
                translate=self.translate,
                save_text=False,
                language=self.language,
                model_name=model,
            )

            # Create sample text with model and voice info
            sample_text = (
                f"{self.sample_text}"
                f"This is a sample using model {model} with voice {voice}. "
            )
            if self.translate:
                sample_text = translate_sentence(
                    sentence=sample_text,
                    src_lang=self.language,
                    tgt_lang=self.translate,
                )

            # Generate audio
            sentences = break_text_into_sentences(sample_text)
            output_wav_path = (
                self.tmp_dir / f"sample_{chapter_index:03d}_{model}_{voice}.wav"
            )

            temp_synthesizer._synthesize_kokoro(sentences, output_wav_path)

            if output_wav_path.exists():
                # Convert to MP3
                mp3_path = temp_synthesizer._convert_to_mp3(output_wav_path)
                logger.info(f"Created sample for {model} + {voice}: {mp3_path}")
                return mp3_path
            else:
                logger.warning(f"Failed to create sample for {model} + {voice}")
                return None

        except Exception as e:
            logger.error(f"Error creating sample for {model} + {voice}: {e}")
            return None

    def _create_metadata_file(
        self, model_voice_combinations: List[Tuple[str, str]]
    ) -> None:
        """Create metadata file for M4B with chapter information."""
        try:
            with open(self.metadata_path, "w") as f:
                f.write(";FFMETADATA1\n")
                f.write("major_brand=M4A\n")
                f.write("minor_version=512\n")
                f.write("compatible_brands=M4A isis2\n")
                f.write("encoder=Lavf61.7.100\n")
                f.write("title=Voice Samples Collection\n")
                f.write("artist=Audify TTS System\n")
                f.write("album=Voice Model Samples\n")
                f.write("genre=Speech Synthesis\n")
                f.write(
                    "comment=Collection of voice samples for all available "
                    "model-voice combinations\n"
                )

            logger.info(f"Created metadata file: {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error creating metadata file: {e}")
            raise

    def _create_m4b_from_samples(self, sample_files: List[Path]) -> None:
        """Create M4B file from sample MP3 files."""
        if not sample_files:
            logger.error("No sample files to create M4B")
            return

        try:
            # Combine all samples into a single M4B
            logger.info(f"Combining {len(sample_files)} samples into M4B...")
            combined_audio = AudioSegment.empty()
            current_start_time_ms = 0

            for i, mp3_file in enumerate(sample_files):
                try:
                    audio = AudioSegment.from_mp3(mp3_file)
                    combined_audio += audio

                    # Extract model and voice from filename
                    filename_parts = mp3_file.stem.split("_")
                    if len(filename_parts) >= 4:
                        model = filename_parts[2]
                        voice = "_".join(filename_parts[3:])
                        chapter_title = f"Model: {model}, Voice: {voice}"
                    else:
                        chapter_title = f"Sample {i + 1}"

                    # Add chapter metadata
                    duration_s = len(audio) / 1000.0
                    self._append_chapter_metadata(
                        current_start_time_ms, duration_s, chapter_title
                    )
                    current_start_time_ms += int(len(audio))

                except Exception as e:
                    logger.error(f"Error processing sample file {mp3_file}: {e}")
                    continue

            if len(combined_audio) == 0:
                logger.error("Combined audio is empty. Cannot create M4B.")
                return

            # Export temporary M4B
            logger.info(
                f"Exporting combined audio to temporary M4B: {self.temp_m4b_path}"
            )
            combined_audio.export(
                self.temp_m4b_path, format="mp4", codec="aac", bitrate="64k"
            )

            # Add metadata using FFmpeg
            self._finalize_m4b()

        except Exception as e:
            logger.error(f"Error creating M4B from samples: {e}")
            raise

    def _append_chapter_metadata(
        self, start_time_ms: int, duration_s: float, title: str
    ) -> None:
        """Append chapter metadata to the metadata file."""
        try:
            with open(self.metadata_path, "a") as f:
                f.write("\n[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={start_time_ms}\n")
                f.write(f"END={start_time_ms + int(duration_s * 1000)}\n")
                f.write(f"title={title}\n")
        except Exception as e:
            logger.error(f"Error appending chapter metadata: {e}")

    def _finalize_m4b(self) -> None:
        """Finalize M4B with metadata using FFmpeg."""
        try:
            logger.info("Adding metadata using FFmpeg...")
            ffmpeg_command = [
                "ffmpeg",
                "-i", str(self.temp_m4b_path),
                "-i", str(self.metadata_path),
                "-map", "0:a",
                "-map_metadata", "1",
                "-c:a", "copy",
                "-f", "mp4",
                "-y", str(self.final_m4b_path),
            ]

            result = subprocess.run(
                ffmpeg_command, check=True, capture_output=True, text=True
            )
            logger.info("FFmpeg process completed successfully.")
            logger.debug(f"FFmpeg stdout: {result.stdout}")
            logger.debug(f"FFmpeg stderr: {result.stderr}")

            # Clean up temporary file
            self.temp_m4b_path.unlink(missing_ok=True)
            logger.info(f"Voice samples M4B created: {self.final_m4b_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {e}")
            logger.error(f"FFmpeg stdout: {e.stdout}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please ensure FFmpeg is installed.")
            raise

    def synthesize(self) -> Path:
        """Create voice samples M4B with all available model-voice combinations."""
        logger.info("Starting voice samples synthesis...")

        # Get available models and voices
        models, voices = self._get_available_models_and_voices()

        if not models or not voices:
            logger.error("No models or voices available. Cannot create samples.")
            return self.final_m4b_path

        # Create all combinations
        model_voice_combinations = [
            (model, voice) for model in models for voice in voices
        ]

        # Limit samples if max_samples is specified
        if self.max_samples and len(model_voice_combinations) > self.max_samples:
            model_voice_combinations = model_voice_combinations[:self.max_samples]
            logger.info(f"Limited to first {self.max_samples} combinations for testing")

        logger.info(
            f"Creating {len(model_voice_combinations)} model-voice combinations"
        )

        # Create metadata file
        self._create_metadata_file(model_voice_combinations)

        # Generate samples for each combination
        sample_files = []
        for i, (model, voice) in enumerate(model_voice_combinations):
            logger.info(
                f"Processing combination {i + 1}/{len(model_voice_combinations)}: "
                f"{model} + {voice}"
            )
            sample_file = self._create_sample_for_combination(model, voice, i + 1)
            if sample_file:
                sample_files.append(sample_file)

        if not sample_files:
            logger.error("No samples were created successfully.")
            return self.final_m4b_path

        # Create M4B from samples
        self._create_m4b_from_samples(sample_files)

        # Clean up temporary files
        try:
            for sample_file in sample_files:
                sample_file.unlink(missing_ok=True)
            if self.tmp_dir.exists():
                shutil.rmtree(self.tmp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")

        logger.info(f"Voice samples synthesis complete: {self.final_m4b_path}")
        return self.final_m4b_path

    def __del__(self):
        """Clean up temporary directory."""
        try:
            if hasattr(self, 'tmp_dir') and self.tmp_dir.exists():
                shutil.rmtree(self.tmp_dir)
        except Exception:
            pass
