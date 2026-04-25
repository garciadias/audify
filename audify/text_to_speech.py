import contextlib
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import requests
from pydub import AudioSegment
from rich.progress import track

from audify.readers.ebook import EpubReader
from audify.readers.pdf import PdfReader
from audify.translate import translate_sentence
from audify.utils.api_config import (
    CommercialAPIConfig,
    OllamaAPIConfig,
    TTSAPIConfig,
    get_tts_config,
)
from audify.utils.audio import AudioProcessor
from audify.utils.constants import (
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    DEFAULT_TTS_PROVIDER,
    KOKORO_API_BASE_URL,
    OUTPUT_BASE_DIR,
)
from audify.utils.logging_utils import setup_logging
from audify.utils.m4b_builder import (
    append_chapter_metadata,
    assemble_m4b,
    write_metadata_header,
)
from audify.utils.text import break_text_into_sentences, get_file_name_title

# Configure logging
logger = setup_logging(module_name=__name__)

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
    """Base class for text-to-speech synthesis.

    This class provides common functionality for synthesizing text into speech.
    Subclasses should implement the `synthesize` method for specific input types.

    Attributes:
    -----------
    path: str | Path
        Path to the audio file
    voice: str
        Voice model to use for synthesis, default is "af_bella",
        to see available voices, run: `audify --list-voices`
    translate: Optional[str]
        Target language for translation before synthesis.
        If None, no translation is done.
    save_text: bool
        Whether to save the original text. Defaults to False.
    language: str
        Language code for synthesis. Defaults to "en".
        To see available languages, run: `audify --list-languages`
    model_name: str
        Model name for synthesis. To see available models, run: `audify --list-models`
    tts_provider: str
        TTS provider to use. Options: "kokoro", "openai", "aws", "google".
        Defaults to DEFAULT_TTS_PROVIDER from environment or "kokoro".

    """

    def __init__(
        self,
        path: str | Path,
        voice: str,
        translate: Optional[str],
        save_text: bool,
        language: str = "en",
        model_name: str = DEFAULT_MODEL,
        tts_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        self.path = Path(path).resolve()
        self.language = language
        self.speaker = voice
        self.translate = translate
        self.model_name = model_name
        self.save_text = save_text
        self.tts_provider = tts_provider or DEFAULT_TTS_PROVIDER
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.tmp_dir_context = tempfile.TemporaryDirectory(
            prefix=f"audify_{self.path.stem}_"
        )
        self.tmp_dir = Path(self.tmp_dir_context.name)

        # Initialize TTS configuration
        self._tts_config: Optional[TTSAPIConfig] = None

    def _get_tts_config(self) -> TTSAPIConfig:
        """Get or create the TTS configuration for the current provider."""
        if self._tts_config is None:
            # Use the target language for synthesis (translated language if translating)
            synthesis_language = self.translate or self.language
            self._tts_config = get_tts_config(
                provider=self.tts_provider,
                voice=self.speaker,
                language=synthesis_language,
            )
            logger.info(
                f"Initialized TTS provider: {self._tts_config.provider_name} "
                f"with voice: {self._tts_config.voice}"
            )
        return self._tts_config

    def _batch_sentences(
        self, sentences: List[str], max_length: int
    ) -> List[List[str]]:
        """Group sentences into batches where each batch's total character
        length <= max_length."""
        batches: List[List[str]] = []
        current_batch: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if sentence_len > max_length:
                # Sentence already exceeds limit; should not happen if
                # sentences are pre-split
                # Add as its own batch
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_length = 0
                batches.append([sentence])
                continue

            separator_len = 1 if current_batch else 0
            if current_length + sentence_len + separator_len <= max_length:
                current_batch.append(sentence)
                current_length += sentence_len + separator_len
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [sentence]
                current_length = sentence_len

        if current_batch:
            batches.append(current_batch)
        return batches

    def _synthesize_with_provider(
        self, sentences: List[str], output_wav_path: Path
    ) -> None:
        """Synthesize sentences using the configured TTS provider."""
        tts_config = self._get_tts_config()
        temp_audio_files: List[Path] = []
        attempted_sentences = 0
        failed_sentences = 0
        consecutive_failures = 0
        _MAX_CONSECUTIVE_FAILURES = 5
        _RECOVERY_WAIT_SECONDS = 15

        try:
            logger.info(f"Starting {tts_config.provider_name} TTS synthesis...")

            # Check if provider is available
            if not tts_config.is_available():
                raise RuntimeError(
                    f"TTS provider '{tts_config.provider_name}' is not available. "
                    "Please check your configuration and credentials."
                )

            # Check voice availability (for providers that support it)
            available_voices = tts_config.get_available_voices()
            if available_voices and tts_config.voice not in available_voices:
                logger.warning(
                    f"Voice '{tts_config.voice}' may not be available for "
                    f"{tts_config.provider_name}."
                    f"Available voices: {available_voices[:5]}..."
                )

            # Group sentences into batches based on provider's max text length
            batches = self._batch_sentences(sentences, tts_config.max_text_length)
            logger.info(
                f"Processing {len(batches)} batch(es) for {len(sentences)} sentences"
            )

            for batch_idx, batch_sentences in track(
                enumerate(batches),
                description=f"{tts_config.provider_name.title()} Synthesizing",
                total=len(batches),
            ):
                # Filter out empty sentences within batch
                batch_sentences = [s for s in batch_sentences if s.strip()]
                if not batch_sentences:
                    continue

                attempted_sentences += len(batch_sentences)

                # If we've had many consecutive failures the server may have
                # crashed/restarted.  Wait for it to come back before wasting
                # more requests.
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.warning(
                        f"{consecutive_failures} consecutive synthesis failures. "
                        f"Waiting {_RECOVERY_WAIT_SECONDS}s for TTS server to "
                        "recover..."
                    )
                    time.sleep(_RECOVERY_WAIT_SECONDS)

                    if not tts_config.is_available():
                        raise RuntimeError(
                            f"TTS provider '{tts_config.provider_name}' became "
                            f"unavailable after {consecutive_failures} consecutive "
                            "failures and did not recover. Check the TTS server."
                        )
                    logger.info("TTS server recovered, resuming synthesis.")
                    consecutive_failures = 0

                try:
                    temp_wav_path = self.tmp_dir / f"batch_{batch_idx}.wav"
                    # Concatenate sentences with a space
                    batch_text = " ".join(batch_sentences)
                    success = tts_config.synthesize(batch_text, temp_wav_path)

                    if success and temp_wav_path.exists():
                        temp_audio_files.append(temp_wav_path)
                        consecutive_failures = 0
                    else:
                        failed_sentences += len(batch_sentences)
                        consecutive_failures += 1
                        logger.warning(
                            f"Synthesis failed for batch {batch_idx}, skipping."
                        )

                except Exception as e:
                    failed_sentences += len(batch_sentences)
                    consecutive_failures += 1
                    logger.warning(f"Error synthesizing batch {batch_idx}: {e}")
                    continue

            logger.info(f"Combining {len(temp_audio_files)} audio segments...")
            try:
                AudioProcessor.combine_wav_segments(
                    temp_audio_files, output_wav_path, logger_instance=logger
                )
            except ValueError as exc:
                if "Combined WAV segments are empty" in str(exc):
                    raise RuntimeError(
                        "No valid audio segments were synthesized. "
                        f"Attempted {attempted_sentences} sentence(s), "
                        f"failed {failed_sentences}. Check TTS provider logs, "
                        "voice/language compatibility, and credentials."
                    ) from exc
                raise

        except Exception as e:
            logger.error(
                f"Error during {tts_config.provider_name} synthesis: {e}",
                exc_info=True,
            )
            for temp_file in temp_audio_files:
                temp_file.unlink(missing_ok=True)
            raise

    def _synthesize_sentences(
        self, sentences: List[str], output_wav_path: Path
    ) -> None:
        """Synthesize a list of sentences into a single WAV file."""
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        self._synthesize_with_provider(sentences, output_wav_path)
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
        output_dir: Optional[str | Path] = None,
        tts_provider: Optional[str] = None,
        llm_config: Optional[Union[OllamaAPIConfig, CommercialAPIConfig]] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        self.reader = EpubReader(path, llm_config=llm_config)
        detected_language = self.reader.get_language()
        language = language or detected_language
        self.output_base_dir = Path(output_dir or OUTPUT_BASE_DIR).resolve()
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
        if not language:
            raise ValueError(
                "Language must be provided or detectable from the EPUB metadata."
            )

        self.title = self.reader.title
        if translate:
            logger.info(f"Translating title from {language} to {translate}")
            self.title = translate_sentence(
                sentence=self.title,
                src_lang=language,
                tgt_lang=translate,
                model=llm_model,
                base_url=llm_base_url,
            )

        self.file_name = get_file_name_title(self.title)
        self._setup_paths(self.file_name)
        self._initialize_metadata_file()
        self.confirm = confirm

        super().__init__(
            path=path,
            language=language,
            voice=speaker,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
            tts_provider=tts_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
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
        write_metadata_header(self.list_of_contents_path)

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
                        sentence,
                        src_lang=self.language,
                        tgt_lang=self.translate,
                        model=self.llm_model,
                        base_url=self.llm_base_url,
                    )
                    for sentence in track(
                        sentences,
                        description=f"Translating Ch. {chapter_number:03d}",
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
        self, chapter_mp3_files: List[Path], max_hours: float = 6.0
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
        AudioProcessor.combine_audio_files(
            chunk_files,
            chunk_temp_path,
            output_format="mp4",
            description=f"Combining Chunk {chunk_index + 1}",
        )
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
        write_metadata_header(chunk_metadata_path)
        current_start_ms = 0
        for mp3_file in chunk_files:
            try:
                chapter_num = int(mp3_file.stem.split("_")[1])
                duration = AudioProcessor.get_duration(str(mp3_file))
                if duration > 0:
                    current_start_ms = append_chapter_metadata(
                        chunk_metadata_path,
                        f"Chapter {chapter_num}",
                        current_start_ms,
                        duration,
                    )
            except Exception as e:
                logger.warning(f"Could not process metadata for {mp3_file}: {e}")
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

        # If duration is less than 6 hours, create a single M4B
        # Using 6 hours as a safe limit to avoid WAV 4GB file size issues
        if total_duration_hours <= 6.0:
            logger.info("Creating single M4B file (duration <= 6 hours)")
            self._create_single_m4b(chapter_mp3_files)
        else:
            logger.info(
                f"Duration ({total_duration_hours:.2f}h) exceeds 6 hours, "
                f"splitting into multiple M4B files to avoid WAV file size limits"
            )
            self._create_multiple_m4bs(chapter_mp3_files)

    def _create_single_m4b(self, chapter_mp3_files: List[Path]) -> None:
        """Create a single M4B file from all chapters."""
        if not self.temp_m4b_path.exists():
            logger.info(f"Combining {len(chapter_mp3_files)} chapter MP3s...")
            AudioProcessor.combine_audio_files(
                chapter_mp3_files,
                self.temp_m4b_path,
                output_format="mp4",
                description="Combining Chapters",
            )
        else:
            logger.info(
                f"Temporary M4B already exists: {self.temp_m4b_path}. Skipping."
            )

        logger.info("Adding metadata and cover image using FFmpeg...")
        assemble_m4b(
            self.temp_m4b_path,
            self.list_of_contents_path,
            self.final_m4b_path,
            getattr(self, "cover_image_path", None),
        )

    def _create_multiple_m4bs(self, chapter_mp3_files: List[Path]) -> None:
        """Create multiple M4B files by splitting chapters into chunks."""
        chunks = self._split_chapters_by_duration(chapter_mp3_files, max_hours=6.0)
        logger.info(f"Split into {len(chunks)} chunks")

        for chunk_index, chunk_files in enumerate(chunks):
            chunk_duration = self._calculate_total_duration(chunk_files) / 3600
            logger.info(
                f"Processing chunk {chunk_index + 1}/{len(chunks)} "
                f"({len(chunk_files)} chapters, {chunk_duration:.2f}h)"
            )

            chunk_temp_path = self._create_temp_m4b_for_chunk(chunk_files, chunk_index)
            if not chunk_temp_path.exists():
                logger.error(
                    f"Failed to create temporary M4B for chunk {chunk_index + 1}"
                )
                continue

            chunk_metadata_path = self._create_metadata_for_chunk(
                chunk_files, chunk_index
            )
            chunk_final_path = (
                self.audiobook_path / f"{self.file_name}_part{chunk_index + 1}.m4b"
            )
            logger.info(
                f"Creating final M4B for chunk {chunk_index + 1}: {chunk_final_path}"
            )
            try:
                assemble_m4b(
                    chunk_temp_path,
                    chunk_metadata_path,
                    chunk_final_path,
                    getattr(self, "cover_image_path", None),
                )
                chunk_metadata_path.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"M4B creation failed for chunk {chunk_index + 1}: {e}")

        logger.info(f"Created {len(chunks)} M4B files for long audiobook")

    def _log_chapter_metadata(
        self, title: str, start_time_ms: int, duration_s: float
    ) -> int:
        """Appends chapter metadata to the FFmpeg metadata file."""
        return append_chapter_metadata(
            self.list_of_contents_path, title, start_time_ms, duration_s
        )

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
            track(chapters, description="Processing Chapters")
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
        output_dir: Optional[str | Path] = None,
        file_name: Optional[str] = None,
        translate: Optional[str] = None,
        save_text: bool = False,
        tts_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        output_dir_path = Path(output_dir or OUTPUT_BASE_DIR).resolve()
        output_dir_path.mkdir(parents=True, exist_ok=True)

        output_base_name = file_name or pdf_path.stem
        self.output_wav_path = output_dir_path / f"{output_base_name}.wav"

        super().__init__(
            path=pdf_path,
            language=language,
            voice=speaker,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
            tts_provider=tts_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
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
                            sentence,
                            src_lang=self.language,
                            tgt_lang=self.translate,
                            model=self.llm_model,
                            base_url=self.llm_base_url,
                        )
                        for sentence in track(sentences, description="Translating PDF")
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
        output_dir: Optional[str | Path] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        self.language = language
        self.translate = translate
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.max_samples = max_samples
        self.sample_text = sample_text or (
            "Bean on bread is a simple yet delightful snack."
            "Hello, this is a sample of the text-to-speech synthesis. "
            "This sample demonstrates the quality and characteristics "
            "of this voice model combination. Each chapter in this audiobook "
            "represents a different voice and model pairing available."
        )

        # Set up paths
        base_dir = Path(output_dir or OUTPUT_BASE_DIR)
        self.output_path = base_dir / "voice_samples"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Set up temporary directory
        self.tmp_dir = Path(tempfile.mkdtemp())

        # Set up M4B paths
        self.temp_m4b_path = self.output_path / "voice_samples.tmp.m4b"
        self.final_m4b_path = self.output_path / "voice_samples.m4b"
        self.metadata_path = self.output_path / "voice_samples_metadata.txt"

    def _get_available_models_and_voices(self) -> Tuple[List[str], List[str]]:
        """Get available models and voices from Kokoro API."""
        from audify.utils.api_config import KokoroAPIConfig, _retry_request

        api_config = KokoroAPIConfig()

        def _fetch():
            models_response = requests.get(
                f"{KOKORO_API_BASE_URL}/models", timeout=10
            )
            models_response.raise_for_status()
            models_data = models_response.json().get("data", [])
            models = sorted(
                [m.get("id") for m in models_data if "id" in m]
            )

            voices_response = requests.get(
                api_config.voices_url, timeout=10
            )
            voices_response.raise_for_status()
            voices_data = voices_response.json().get("voices", [])
            voices = sorted(voices_data)

            logger.info(
                f"Found {len(models)} models and {len(voices)} voices"
            )
            return models, voices

        try:
            return _retry_request(
                _fetch,
                api_name=f"Kokoro API ({KOKORO_API_BASE_URL})",
            )
        except RuntimeError:
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
                    model=self.llm_model,
                    base_url=self.llm_base_url,
                )

            # Generate audio
            sentences = break_text_into_sentences(sample_text)
            output_wav_path = (
                self.tmp_dir / f"sample_{chapter_index:03d}_{model}_{voice}.wav"
            )

            temp_synthesizer._synthesize_sentences(sentences, output_wav_path)

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
        write_metadata_header(self.metadata_path)
        # Append extra tags not included in the standard header
        try:
            with open(self.metadata_path, "a") as f:
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

                    # Extract model and voice from file_name
                    file_name_parts = mp3_file.stem.split("_")
                    if len(file_name_parts) >= 4:
                        model = file_name_parts[2]
                        voice = "_".join(file_name_parts[3:])
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
        logger.info("Adding metadata using FFmpeg...")
        assemble_m4b(
            self.temp_m4b_path,
            self.metadata_path,
            self.final_m4b_path,
            cover_image=None,
        )
        logger.info(f"Voice samples M4B created: {self.final_m4b_path}")

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
            model_voice_combinations = model_voice_combinations[: self.max_samples]
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
            if hasattr(self, "tmp_dir") and self.tmp_dir.exists():
                shutil.rmtree(self.tmp_dir)
        except Exception:
            pass
