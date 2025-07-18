"""
Enhanced synthesizers with progress monitoring and cancellation support.
"""
import contextlib
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from typing_extensions import Literal

from audify.text_to_speech import DEFAULT_MODEL, DEFAULT_SPEAKER, BaseSynthesizer
from audify.text_to_speech import EpubSynthesizer as OriginalEpubSynthesizer
from audify.text_to_speech import PdfSynthesizer as OriginalPdfSynthesizer
from audify.translate import translate_sentence
from audify.utils import break_text_into_sentences

logger = logging.getLogger(__name__)


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


class EnhancedBaseSynthesizer(BaseSynthesizer):
    """Enhanced base synthesizer with progress tracking and cancellation."""

    def __init__(
        self,
        path: str | Path,
        language: Optional[str],
        speaker: str,
        model_name: str,
        translate: Optional[str],
        save_text: bool,
        engine: str,
        progress_callback: Optional[Callable] = None,
        cancellation_check: Optional[Callable] = None,
    ):
        # Initialize progress tracking
        self.progress_callback = progress_callback or (lambda **kwargs: None)
        self.cancellation_check = cancellation_check or (lambda: False)

        # Call parent constructor
        super().__init__(
            path, language, speaker, model_name, translate, save_text, engine
        )

    def _check_cancellation(self):
        """Check if the job has been cancelled."""
        if self.cancellation_check():
            raise InterruptedError("Job was cancelled")

    def _update_progress(self, **kwargs):
        """Update progress information."""
        try:
            self.progress_callback(**kwargs)
        except Exception as e:
            logger.warning(f"Error updating progress: {e}")

    def _convert_to_mp3(self, wav_path: Path) -> Path:
        """Converts a WAV file to MP3 and removes the original WAV."""
        from pydub import AudioSegment

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

    def _synthesize_sentences_with_progress(
        self, sentences: List[str], output_wav_path: Path, operation_name: str = "Synthesizing"
    ) -> None:
        """Synthesize sentences with progress tracking and cancellation support."""
        self._update_progress(
            current_step=0,
            total_steps=len(sentences),
            current_operation=f"{operation_name} audio..."
        )

        if self.engine == "kokoro":
            self._synthesize_kokoro_with_progress(sentences, output_wav_path, operation_name)
        elif self.engine == "tts_models":
            self._synthesize_tts_models_with_progress(sentences, output_wav_path, operation_name)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    def _synthesize_kokoro_with_progress(
        self, sentences: List[str], output_wav_path: Path, operation_name: str
    ) -> None:
        """Synthesize using Kokoro with progress tracking."""
        from kokoro import KPipeline
        from pydub import AudioSegment

        synthesis_language = self.translate or self.language
        lang_code = self._get_lang_code(synthesis_language)

        kokoro = KPipeline(self.device)
        combined_audio = AudioSegment.empty()
        temp_file = self.tmp_dir / "temp_synthesis.wav"

        total_sentences = len([s for s in sentences if s.strip()])
        processed_sentences = 0

        for i, sentence in enumerate(sentences):
            self._check_cancellation()

            if not sentence.strip():
                continue

            self._update_progress(
                current_step=i + 1,
                total_steps=len(sentences),
                current_operation=f"{operation_name} sentence {processed_sentences + 1}/{total_sentences}"
            )

            try:
                audio_data, sample_rate = kokoro(sentence, lang_code, speed=1.0)

                # Save temporary audio
                import soundfile as sf
                sf.write(temp_file, audio_data, sample_rate)

                if temp_file.exists():
                    segment = AudioSegment.from_wav(temp_file)
                    combined_audio += segment
                    temp_file.unlink()

                processed_sentences += 1

            except Exception as e:
                logger.error(f"Error synthesizing sentence: '{sentence[:50]}...'. Error: {e}")
                continue

        logger.info(f"Exporting combined audio to {output_wav_path}")
        combined_audio.export(output_wav_path, format="wav")

    def _synthesize_tts_models_with_progress(
        self, sentences: List[str], output_wav_path: Path, operation_name: str
    ) -> None:
        """Synthesize using TTS models with progress tracking."""
        from pydub import AudioSegment

        combined_audio = AudioSegment.empty()
        synthesis_lang = self.translate or self.language
        temp_speech_path = self.tmp_dir / "speech_segment.wav"

        total_sentences = len([s for s in sentences if s.strip()])
        processed_sentences = 0

        for i, sentence in enumerate(sentences):
            self._check_cancellation()

            if not sentence.strip():
                continue

            self._update_progress(
                current_step=i + 1,
                total_steps=len(sentences),
                current_operation=f"{operation_name} sentence {processed_sentences + 1}/{total_sentences}"
            )

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

                processed_sentences += 1

            except Exception as e:
                logger.error(f"Error synthesizing sentence: '{sentence[:50]}...'. Error: {e}")
                if temp_speech_path.exists():
                    temp_speech_path.unlink(missing_ok=True)
                continue

        logger.info(f"Exporting combined TTS audio to {output_wav_path}")
        combined_audio.export(output_wav_path, format="wav")

    def _get_lang_code(self, language: str) -> str:
        """Get language code for Kokoro engine."""
        from audify.constants import LANG_CODES
        return LANG_CODES.get(language, "a")


class EnhancedEpubSynthesizer(EnhancedBaseSynthesizer):
    """Enhanced EPUB synthesizer with progress tracking and cancellation."""

    def __init__(
        self,
        path: str | Path,
        language: Optional[str] = None,
        speaker: str = DEFAULT_SPEAKER,
        model_name: str = DEFAULT_MODEL,
        translate: Optional[str] = None,
        save_text: bool = False,
        engine: Literal["kokoro", "tts_models"] = "kokoro",
        confirm: bool = True,
        progress_callback: Optional[Callable] = None,
        cancellation_check: Optional[Callable] = None,
    ):
        # Initialize the original EPUB synthesizer to get the setup
        self._original = OriginalEpubSynthesizer(
            path, language, speaker, model_name, translate, save_text, engine, confirm
        )

        # Copy necessary attributes
        self.reader = self._original.reader
        self.title = self._original.title
        self.file_name = self._original.file_name
        self.audiobook_path = self._original.audiobook_path
        self.temp_m4b_path = self._original.temp_m4b_path
        self.final_m4b_path = self._original.final_m4b_path
        self.chapters_file_path = self._original.chapters_file_path
        self.metadata_txt_path = self._original.metadata_txt_path
        self.cover_image_path = self._original.cover_image_path

        # Initialize enhanced base with progress support
        super().__init__(
            path, self._original.language, speaker, model_name, translate, save_text, engine,
            progress_callback, cancellation_check
        )

    def synthesize(self) -> Path:
        """Synthesize EPUB to audiobook with progress tracking."""
        try:
            self._check_cancellation()

            chapters = self.reader.get_chapters()
            total_chapters = len(chapters)

            self._update_progress(
                current_step=0,
                total_steps=total_chapters,
                current_operation="Starting synthesis..."
            )

            logger.info(f"Starting synthesis of {total_chapters} chapters...")

            # Process each chapter
            for chapter_num, chapter_text in enumerate(chapters, 1):
                self._check_cancellation()

                self._update_progress(
                    current_step=chapter_num - 1,
                    total_steps=total_chapters,
                    current_operation=f"Processing chapter {chapter_num}/{total_chapters}..."
                )

                chapter_mp3_path = self._synthesize_chapter_with_progress(
                    chapter_text, chapter_num, total_chapters
                )

                logger.info(f"Completed chapter {chapter_num}: {chapter_mp3_path}")

            self._check_cancellation()

            # Create final M4B file
            self._update_progress(
                current_step=total_chapters,
                total_steps=total_chapters + 1,
                current_operation="Creating final audiobook..."
            )

            self._create_m4b_with_progress()

            self._update_progress(
                current_step=total_chapters + 1,
                total_steps=total_chapters + 1,
                current_operation="Synthesis complete!"
            )

            logger.info(f"EPUB synthesis complete: {self.final_m4b_path}")
            return self.final_m4b_path

        except InterruptedError:
            logger.info("EPUB synthesis was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error during EPUB synthesis: {e}", exc_info=True)
            raise

    def _synthesize_chapter_with_progress(
        self, chapter_txt: str, chapter_number: int, total_chapters: int
    ) -> Path:
        """Synthesize a single chapter with progress tracking."""
        chapter_wav_path = self.audiobook_path / f"chapter_{chapter_number:03d}.wav"

        # Break into sentences
        sentences = break_text_into_sentences(chapter_txt)

        # Handle translation if needed
        if self.translate and self.language:
            self._update_progress(
                current_operation=f"Translating chapter {chapter_number}/{total_chapters}..."
            )

            try:
                sentences = [
                    translate_sentence(
                        sentence, src_lang=self.language, tgt_lang=self.translate
                    )
                    for sentence in sentences
                ]
            except Exception as e:
                logger.error(f"Error translating chapter {chapter_number}: {e}")
                sentences = break_text_into_sentences(chapter_txt)

        # Synthesize sentences
        self._synthesize_sentences_with_progress(
            sentences, chapter_wav_path, f"Chapter {chapter_number}"
        )

        # Convert to MP3
        return self._convert_to_mp3(chapter_wav_path)

    def _create_m4b_with_progress(self) -> None:
        """Create M4B file with progress tracking."""
        from pydub import AudioSegment

        chapter_mp3_files = sorted(self.audiobook_path.glob("chapter_*.mp3"))

        if not chapter_mp3_files:
            raise ValueError("No chapter MP3 files found to create M4B.")

        self._update_progress(current_operation="Combining chapters into audiobook...")

        combined_audio = AudioSegment.empty()
        for i, mp3_file in enumerate(chapter_mp3_files):
            self._check_cancellation()

            try:
                audio = AudioSegment.from_mp3(mp3_file)
                combined_audio += audio

                self._update_progress(
                    current_operation=f"Combining chapters... ({i + 1}/{len(chapter_mp3_files)})"
                )

            except Exception as e:
                logger.error(f"Error processing chapter file {mp3_file}: {e}")
                continue

        # Export combined audio
        logger.info(f"Exporting combined audiobook to {self.temp_m4b_path}")
        combined_audio.export(self.temp_m4b_path, format="mp3")

        # Copy metadata and finalize
        import shutil
        shutil.copy2(self.temp_m4b_path, self.final_m4b_path)


class EnhancedPdfSynthesizer(EnhancedBaseSynthesizer):
    """Enhanced PDF synthesizer with progress tracking and cancellation."""

    def __init__(
        self,
        path: str | Path,
        language: Optional[str] = None,
        speaker: str = DEFAULT_SPEAKER,
        model_name: str = DEFAULT_MODEL,
        translate: Optional[str] = None,
        save_text: bool = False,
        engine: Literal["kokoro", "tts_models"] = "kokoro",
        progress_callback: Optional[Callable] = None,
        cancellation_check: Optional[Callable] = None,
    ):
        # Initialize the original PDF synthesizer to get the setup
        self._original = OriginalPdfSynthesizer(
            path, language, speaker, model_name, translate, save_text, engine
        )

        # Copy necessary attributes
        self.reader = self._original.reader
        self.output_wav_path = self._original.output_wav_path

        # Initialize enhanced base with progress support
        super().__init__(
            path, language or "en", speaker, model_name, translate, save_text, engine,
            progress_callback, cancellation_check
        )

    def synthesize(self) -> Path:
        """Synthesize PDF to audio with progress tracking."""
        try:
            self._check_cancellation()

            self._update_progress(
                current_step=0,
                total_steps=4,
                current_operation="Extracting text from PDF..."
            )

            # Extract text
            cleaned_text = self.reader.get_text()

            if not cleaned_text.strip():
                raise ValueError("No text extracted from PDF. Cannot synthesize.")

            self._check_cancellation()

            # Break into sentences
            self._update_progress(
                current_step=1,
                total_steps=4,
                current_operation="Processing text..."
            )

            sentences = break_text_into_sentences(cleaned_text)

            if not sentences:
                raise ValueError("No sentences extracted from PDF. Cannot synthesize.")

            logger.info(f"Extracted {len(sentences)} sentences.")

            self._check_cancellation()

            # Handle translation if needed
            if self.translate and self.language:
                self._update_progress(
                    current_step=2,
                    total_steps=4,
                    current_operation="Translating text..."
                )

                try:
                    sentences = [
                        translate_sentence(
                            sentence, src_lang=self.language, tgt_lang=self.translate
                        )
                        for sentence in sentences
                    ]
                except Exception as e:
                    logger.error(f"Error translating PDF content: {e}")
                    sentences = break_text_into_sentences(cleaned_text)

            self._check_cancellation()

            # Synthesize audio
            self._update_progress(
                current_step=3,
                total_steps=4,
                current_operation="Synthesizing audio..."
            )

            self.output_wav_path.parent.mkdir(parents=True, exist_ok=True)
            self._synthesize_sentences_with_progress(sentences, self.output_wav_path, "PDF")

            # Convert to MP3
            self._update_progress(
                current_step=4,
                total_steps=4,
                current_operation="Converting to MP3..."
            )

            final_mp3_path = self._convert_to_mp3(self.output_wav_path)

            logger.info(f"PDF synthesis complete: {final_mp3_path}")
            return final_mp3_path

        except InterruptedError:
            logger.info("PDF synthesis was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error during PDF synthesis: {e}", exc_info=True)
            raise
