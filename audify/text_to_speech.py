import contextlib
import logging
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

import soundfile as sf
import torch
import tqdm
from kokoro import KPipeline
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from TTS.api import TTS
from typing_extensions import Literal

from audify.constants import LANG_CODES
from audify.domain.interface import Synthesizer
from audify.ebook_read import EpubReader
from audify.pdf_read import PdfReader
from audify.translate import translate_sentence
from audify.utils import (
    break_text_into_sentences,
    get_audio_duration,
    get_file_name_title,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODULE_PATH = Path(__file__).resolve().parents[1]
DEFAULT_SPEAKER = "data/Jennifer_16khz.wav"
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_ENGINE = "kokoro"
OUTPUT_BASE_DIR = MODULE_PATH / "../" / "data" / "output"

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


class BaseSynthesizer(Synthesizer):
    """Base class for text-to-speech synthesis."""

    def __init__(
        self,
        path: str | Path,
        language: Optional[str],
        speaker: str,
        model_name: str | None,
        translate: Optional[str],
        save_text: bool,
        engine: str,
    ):
        self.path = Path(path).resolve()
        self.language = language
        self.speaker = speaker
        self.translate = translate
        self.engine = engine
        self.model_name = model_name
        self.save_text = save_text
        self.tmp_dir_context = tempfile.TemporaryDirectory(
            prefix=f"audify_{self.path.stem}_"
        )
        self.tmp_dir = Path(self.tmp_dir_context.name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        if engine == "tts_models":
            logger.info(f"Loading TTS model: {model_name}...")
            self.model = TTS(model_name=model_name)
            self.model.to(self.device)
            logger.info("TTS model loaded.")

    def _synthesize_kokoro(self, sentences: List[str], output_wav_path: Path) -> None:
        """Synthesize sentences using the Kokoro engine."""
        synthesis_language = self.translate or self.language
        if synthesis_language is None:
            logger.warning(
                "Language not specified or detected, defaulting to 'a' for Kokoro."
            )
            lang_code = "a"
        else:
            lang_code = LANG_CODES.get(synthesis_language, "a")

        if not hasattr(self, "pipeline") or self.pipeline.lang_code != lang_code:
            logger.info(f"Initializing Kokoro pipeline for language: {lang_code}")
            self.pipeline: KPipeline = KPipeline(lang_code=lang_code)

        combined_audio = AudioSegment.empty()
        temp_audio_files: List[Path] = []

        try:
            logger.info("Starting Kokoro synthesis...")
            generator: Generator[Tuple[int, str, Any], None, None] = self.pipeline(
                sentences,
                voice="af_heart",
                speed=1,
                split_pattern=r"\n+",
            )

            for i, (_, _, audio_data) in tqdm.tqdm(
                enumerate(generator),
                desc="Kokoro Synthesizing",
                total=len(sentences),
                unit="sentence",
            ):
                temp_wav_path = self.tmp_dir / f"segment_{i}.wav"
                sf.write(temp_wav_path, audio_data, 24000)
                temp_audio_files.append(temp_wav_path)

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
            logger.error(f"Error during Kokoro synthesis: {e}", exc_info=True)
            for temp_file in temp_audio_files:
                temp_file.unlink(missing_ok=True)
            raise

    def _synthesize_tts_models(
        self, sentences: List[str], output_wav_path: Path
    ) -> None:
        """Synthesize sentences using the TTS library models."""
        combined_audio = AudioSegment.empty()
        synthesis_lang = self.translate or self.language
        temp_speech_path = self.tmp_dir / "speech_segment.wav"

        logger.info("Starting TTS models synthesis...")
        for sentence in tqdm.tqdm(sentences, desc="TTS Synthesizing", unit="sentence"):
            if not sentence.strip():
                continue
            try:
                with suppress_stdout():
                    self.model.tts_to_file(
                        text=sentence,
                        speaker=self.speaker,
                        language=synthesis_lang,
                        file_path=str(temp_speech_path),
                    )
                if temp_speech_path.exists():
                    segment = AudioSegment.from_wav(temp_speech_path)
                    combined_audio += segment
                    temp_speech_path.unlink()
                else:
                    logger.warning(
                        f"TTS output file not found for sentence: '{sentence[:50]}...'"
                    )
            except Exception as e:
                logger.error(
                    f"Error synthesizing sentence: '{sentence[:50]}...'. Error: {e}",
                    exc_info=False,
                )
                if temp_speech_path.exists():
                    temp_speech_path.unlink(missing_ok=True)

        logger.info(f"Exporting combined TTS audio to {output_wav_path}")
        combined_audio.export(output_wav_path, format="wav")

    def _synthesize_sentences(
        self, sentences: List[str], output_wav_path: Path
    ) -> None:
        """Synthesize a list of sentences into a single WAV file."""
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        if self.engine == "kokoro":
            self._synthesize_kokoro(sentences, output_wav_path)
        elif self.engine == "tts_models":
            self._synthesize_tts_models(sentences, output_wav_path)
        else:
            raise ValueError(f"Unsupported synthesis engine: {self.engine}")

        logger.info(f"Raw WAV synthesis complete: {output_wav_path}")

    def _convert_to_mp3(self, wav_path: Path) -> Path:
        """Converts a WAV file to MP3 and removes the original WAV."""
        mp3_path = wav_path.with_suffix(".mp3")
        logger.info(f"Converting {wav_path.name} to MP3...")
        try:
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3", bitrate="192k")
            logger.info(f"MP3 conversion successful: {mp3_path.name}")
            wav_path.unlink(missing_ok=True)
            return mp3_path
        except FileNotFoundError:
            logger.error(f"WAV file not found for conversion: {wav_path}")
            raise
        except Exception as e:
            logger.error(f"Error converting {wav_path.name} to MP3: {e}", exc_info=True)
            raise

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
        if hasattr(self, "pipeline") and self.pipeline.is_running:
            self.pipeline.stop()
            logger.info("Synthesis process stopped.")
        else:
            logger.warning("No active synthesis process to stop.")

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
        model_name: str | None = DEFAULT_MODEL,
        translate: Optional[str] = None,
        save_text: bool = False,
        engine: Literal["kokoro", "tts_models"] = "kokoro",
        confirm: bool = True,
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
            path, resolved_language, speaker, model_name, translate, save_text, engine
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
        logger.info(f"Synthesizing Chapter {chapter_number}...")
        chapter_wav_path = self.audiobook_path / f"chapter_{chapter_number}.wav"
        chapter_mp3_path = chapter_wav_path.with_suffix(".mp3")

        if chapter_mp3_path.exists():
            logger.info(
                f"Chapter {chapter_number} MP3 already exists, skipping synthesis."
            )
            return chapter_mp3_path

        chapter_txt = self.reader.extract_text(chapter_content)
        sentences = break_text_into_sentences(chapter_txt)

        if not sentences:
            logger.warning(f"Chapter {chapter_number} contains no text to synthesize.")
            return chapter_mp3_path

        if self.translate and self.language:
            logger.info(
                f"Translating {len(sentences)} sentences for Chapter"
                f" {chapter_number}..."
            )
            try:
                sentences = [
                    translate_sentence(
                        sentence, src_lang=self.language, tgt_lang=self.translate
                    )
                    for sentence in tqdm.tqdm(
                        sentences,
                        desc=f"Translating Ch. {chapter_number}",
                        unit="sentence",
                    )
                ]
            except Exception as e:
                logger.error(
                    f"Error translating chapter {chapter_number}: {e}", exc_info=True
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
            try:
                duration = get_audio_duration(str(mp3_file))
                total_duration += duration
            except Exception as e:
                logger.warning(f"Could not get duration for {mp3_file}: {e}")
        return total_duration

    def _split_chapters_by_duration(
        self, chapter_mp3_files: List[Path], max_hours: float = 15.0
    ) -> List[List[Path]]:
        """Split chapter MP3 files into chunks with maximum duration in hours."""
        max_duration_seconds = max_hours * 3600  # Convert hours to seconds
        chunks = []
        current_chunk: list[Path] = []
        current_duration = 0.0

        for mp3_file in chapter_mp3_files:
            try:
                file_duration = get_audio_duration(str(mp3_file))

                # If adding this file would exceed the limit, start a new chunk
                if (
                    current_chunk
                    and (current_duration + file_duration) > max_duration_seconds
                ):
                    chunks.append(current_chunk)
                    current_chunk = [mp3_file]
                    current_duration = file_duration
                else:
                    current_chunk.append(mp3_file)
                    current_duration += file_duration

            except Exception as e:
                logger.warning(
                    f"Could not get duration for {mp3_file}: {e}, "
                    f"adding to current chunk anyway"
                )
                current_chunk.append(mp3_file)

        # Add the last chunk if it has files
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

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
                        duration = get_audio_duration(str(mp3_file))

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

        chapter_mp3_path = self.audiobook_path / f"chapter_{chapter_index}.mp3"
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
                duration_s = get_audio_duration(str(chapter_mp3_path))
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
                    duration_s = get_audio_duration(str(synthesized_path))
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

    DEFAULT_SPEAKER = "data/Jennifer_16khz.wav"
    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    DEFAULT_ENGINE = "kokoro"
    DEFAULT_OUTPUT_DIR = MODULE_PATH / "../" / "data" / "output" / "articles"

    def __init__(
        self,
        pdf_path: str | Path,
        language: str = "en",
        model_name: str | None = DEFAULT_MODEL,
        speaker: str = DEFAULT_SPEAKER,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        file_name: Optional[str] = None,
        translate: Optional[str] = None,
        save_text: bool = False,
        engine: Literal["kokoro", "tts_models"] = "kokoro",
    ):
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        output_base_name = file_name or pdf_path.stem
        self.output_wav_path = output_dir / f"{output_base_name}.wav"

        super().__init__(
            pdf_path, language, speaker, model_name, translate, save_text, engine
        )

    def synthesize(self) -> Path:
        """Reads PDF, synthesizes text, and returns the path to the MP3."""
        logger.info(f"Starting synthesis for PDF: {self.path.name}")
        try:
            reader = PdfReader(self.path)
            logger.info("Extracting and cleaning text from PDF...")
            cleaned_text = reader.get_cleaned_text()
            sentences = break_text_into_sentences(cleaned_text)

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
                    sentences = break_text_into_sentences(cleaned_text)

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


class InspectSynthesizer(Synthesizer):
    """Utility class to inspect TTS model properties (speakers, languages)."""

    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        path: str | Path = "./",
        language: Optional[str] = None,
        speaker: Optional[str] = None,
    ):
        self.model_name = model_name
        logger.info(f"Loading model {model_name} for inspection...")
        try:
            self.model = TTS(model_name=model_name)
            logger.info("Model loaded successfully for inspection.")
        except Exception as e:
            logger.error(
                f"Failed to load model {model_name} for inspection: {e}", exc_info=True
            )
            raise

    def list_languages(self) -> Optional[List[str]]:
        """Lists available languages for the loaded model."""
        if hasattr(self.model, "languages") and self.model.languages:
            logger.info(
                f"Available languages for {self.model_name}: {self.model.languages}"
            )
            return self.model.languages
        else:
            logger.warning(
                f"Model {self.model_name} does not seem to report languages."
            )
            return None

    def list_speakers(self) -> Optional[List[str]]:
        """Lists available speakers for the loaded model."""
        if self.model.is_multi_speaker:
            if (
                "xtts" in self.model_name.lower()
                and hasattr(self.model, "speaker_manager")
                and hasattr(self.model.speaker_manager, "speaker_ids")
            ):
                speakers = list(self.model.speaker_manager.speaker_ids.keys())
                logger.info(
                    f"Available speakers (IDs/Names) for {self.model_name}: {speakers}"
                )
                return speakers
            elif hasattr(self.model, "speaker_ids") and self.model.speaker_ids:
                speakers = list(self.model.speaker_ids.keys())
                logger.info(
                    f"Available speakers (IDs) for {self.model_name}: {speakers}"
                )
                return speakers
            else:
                logger.warning(
                    f"Could not determine speaker list for multi-speaker model"
                    f" {self.model_name}."
                )
                return None
        else:
            logger.info(f"Model {self.model_name} is single-speaker.")
            return None

    def list_models(self) -> List[str]:
        """Lists available TTS models from the TTS library."""
        logger.info("Fetching list of available TTS models...")
        try:
            available_models = TTS.list_models()
            logger.info(f"Found {len(available_models)} available models.")
            return available_models
        except Exception as e:
            logger.error(
                f"Failed to fetch list of available models: {e}", exc_info=True
            )
            return []

    def synthesize(self) -> str:
        """Indicates that this class is not for direct synthesis."""
        logger.warning("InspectSynthesizer is used for inspection, not synthesis.")
        return (
            "This class is used to inspect model options (languages, speakers, models) "
            "and does not perform audio synthesis."
        )
