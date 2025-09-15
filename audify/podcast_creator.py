import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import tqdm
from langchain_ollama import OllamaLLM
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from audify.constants import OLLAMA_API_BASE_URL, OLLAMA_DEFAULT_MODEL, OUTPUT_BASE_DIR
from audify.ebook_read import EpubReader
from audify.pdf_read import PdfReader
from audify.prompts import PODCAST_PROMPT
from audify.text_to_speech import BaseSynthesizer
from audify.translate import translate_sentence
from audify.utils import clean_text, get_audio_duration

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODULE_PATH = Path(__file__).resolve().parents[1]


class LLMClient:
    """Client for interacting with local LLM using LangChain."""

    def __init__(
        self, base_url: str = OLLAMA_API_BASE_URL, model: str = OLLAMA_DEFAULT_MODEL
    ):
        self.base_url = base_url
        self.model = model

        # Initialize LangChain Ollama client
        self.llm = OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            # LangChain Ollama specific parameters
            num_ctx=8 * 4096,  # Increased context window
            temperature=0.8,  # Added creativity
            top_p=0.9,  # Broader token selection
            repeat_penalty=1.05,  # Slight penalty for repetition
            seed=428798,
            top_k=60,  # Wider token selection
            num_predict=4096,  # Encourage longer responses
        )

    def generate_podcast_script(
        self, chapter_text: str, language: Optional[str]
    ) -> str:
        """Generate podcast script from chapter text using LangChain."""
        if language:
            translated_prompt = translate_sentence(
                PODCAST_PROMPT, src_lang="en", tgt_lang=language
            )
            prompt = translated_prompt + "\n\n" + chapter_text
        else:
            prompt = PODCAST_PROMPT + "\n\n" + chapter_text

        try:
            logger.info(f"Sending request to LLM at {self.base_url}")
            response = self.llm.invoke(prompt)

            # Clean the response
            if response:
                # if it is a reasoning model, eliminate the reasoning steps
                if "think" in response.lower():
                    return clean_text(response.split("</think>")[-1].strip())
                return clean_text(response.strip())
            else:
                logger.error("Empty response from LLM")
                return "Error: Unable to generate podcast script for this content."

        except Exception as e:
            logger.error(f"Error communicating with LLM: {e}")
            if "connection" in str(e).lower():
                return (
                    "Error: Could not connect to local LLM server. "
                    f"Please ensure Ollama is running at {self.base_url}."
                )
            elif "timeout" in str(e).lower():
                return "Error: Request to LLM timed out. Content might be too long."
            else:
                return f"Error: Failed to generate podcast script due to: {str(e)}"


class PodcastCreator(BaseSynthesizer):
    """Creates podcasts from ebook content using LLM and TTS."""

    def __init__(
        self,
        path: str | Path,
        language: Optional[str] = None,
        voice: str = "af_bella",
        model_name: str = OLLAMA_DEFAULT_MODEL,
        translate: Optional[str] = None,
        save_text: bool = True,  # Default to True for podcast scripts
        llm_base_url: str = OLLAMA_API_BASE_URL,
        llm_model: str = OLLAMA_DEFAULT_MODEL,
        max_chapters: Optional[int] = None,
        confirm: bool = True,
    ):
        # Initialize file reader based on extension
        file_path = Path(path)
        self.reader: EpubReader | PdfReader = (
            EpubReader(path) if file_path.suffix.lower() == ".epub" else PdfReader(path)
        )
        if isinstance(self.reader, EpubReader):
            detected_language = self.reader.get_language()
            self.title = self.reader.title
        elif isinstance(self.reader, PdfReader):
            detected_language = "en"  # PDF reader might not detect language
            self.title = file_path.stem
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        self.resolved_language = language or detected_language
        if not self.resolved_language:
            raise ValueError(
                "Language must be provided or detectable from the file metadata."
            )

        # Setup output paths
        self.output_base_dir = Path(OUTPUT_BASE_DIR).resolve()
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Create podcast-specific title
        self.podcast_title = f"Podcast - {self.title}"
        # Use pdf filename as the podcast filename
        self.file_name = Path(file_path.stem)
        self._setup_paths(self.file_name)

        # Initialize LLM client
        self.llm_client = LLMClient(llm_base_url, llm_model)
        self.max_chapters = max_chapters
        self.confirm = confirm

        # Initialize parent class
        super().__init__(
            path=path,
            language=self.resolved_language,
            voice=voice,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
        )

        # Setup cover image if available
        if isinstance(self.reader, EpubReader):
            self.cover_image_path: Optional[Path] = self.reader.get_cover_image(
                self.podcast_path
            )
        else:
            self.cover_image_path = None
        self.chapter_titles: List[str] = []

    def _setup_paths(self, file_name_base: Path) -> None:
        """Sets up the necessary output paths for podcast creation."""
        if not self.output_base_dir.exists():
            logger.info(f"Creating output base directory: {self.output_base_dir}")
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.podcast_path = self.output_base_dir / file_name_base
        self.podcast_path.mkdir(parents=True, exist_ok=True)
        self.scripts_path = self.podcast_path / "scripts"
        self.scripts_path.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.podcast_path / "episodes"
        self.episodes_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Podcast output directory set to: {self.podcast_path}")

    def _clean_text_for_podcast(self, text: str) -> str:
        """
        Clean text by removing references, citations, and other non-content elements.
        """
        text = str(text)  # Ensure text is a string
        print("Cleaning text to remove references and citations...")
        print("Original text length:", len(text))
        print("Original text snippet:", text[:500])
        # Remove common reference patterns
        # Remove numbered references like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Remove references with author-year format like (Smith, 2020)
        text = re.sub(r"\([^)]*\d{4}[^)]*\)", "", text)

        # Remove standalone citations like "Smith et al. (2020)"
        text = re.sub(r"\b[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)", "", text)

        # Remove DOI patterns
        text = re.sub(r"doi:\s*[\d\.\w/\-]+", "", text)

        # Remove URL patterns
        text = re.sub(r"http[s]?://[\w\.\-/\?\=&%]+", "", text)
        text = re.sub(r"www\.[\w\.\-/\?\=&%]+", "", text)

        # Remove common reference section headers and content
        # Match "References", "Bibliography", "Works Cited" and similar
        reference_patterns = [
            r"(?i)references?\s*:?\s*\n.*?(?=\n\s*[A-Z]|\Z)",
            r"(?i)bibliography\s*:?\s*\n.*?(?=\n\s*[A-Z]|\Z)",
            r"(?i)works?\s+cited\s*:?\s*\n.*?(?=\n\s*[A-Z]|\Z)",
            r"(?i)literature\s+cited\s*:?\s*\n.*?(?=\n\s*[A-Z]|\Z)",
        ]

        for pattern in reference_patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL)

        # Remove common academic formatting
        # Remove figure/table references like "Figure 1", "Table 2", etc.
        text = re.sub(r"(?i)\b(figure|fig|table|tab)\s*\.?\s*\d+", "", text)

        # Remove page numbers and similar formatting
        text = re.sub(r"\bpp?\.\s*\d+(-\d+)?", "", text)

        # Clean up multiple spaces and normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        return text.strip()

    def generate_podcast_script(
        self, chapter_text: str, chapter_number: int, language: Optional[str]
    ) -> str:
        """Generate podcast script for a single chapter."""

        logger.info(f"Generating podcast script for Chapter {chapter_number}...")

        # Save script path
        script_path = self.scripts_path / f"episode_{chapter_number:03d}_script.txt"
        # Check if script already exists
        if script_path.exists() and not self.confirm:
            logger.info(
                f"Script for Episode {chapter_number} already exists, loading..."
            )
            with open(script_path, "r", encoding="utf-8") as f:
                return f.read()

        if not chapter_text.strip():
            logger.warning(f"No text found in Chapter {chapter_number}")
            return "This chapter contains no readable text content."

        # Clean text to remove references and citations
        cleaned_text = self._clean_text_for_podcast(chapter_text)
        logger.info(
            f"Cleaned text for Episode {chapter_number}:"
            " removed references and citations"
        )
        logger.info(
            f"Original length: {len(chapter_text.split())} words, "
            f"Cleaned length: {len(cleaned_text.split())} words"
        )
        chapter_title = (
            self.reader.get_chapter_title(chapter_text)
            if isinstance(self.reader, EpubReader)
            else self.reader.path.stem
        )
        logger.info(f"Chapter {chapter_number} title: {chapter_title}")
        self.chapter_titles.append(chapter_title)

        # Generate podcast script using LLM
        if len(cleaned_text.split()) < 200:
            logger.warning(
                f"Chapter {chapter_number} has very little text after cleaning. "
                "The generated podcast may be very short."
            )
            podcast_script = chapter_text
        else:
            if self.translate and language:
                translated_prompt = translate_sentence(
                    PODCAST_PROMPT, src_lang="en", tgt_lang=language
                )
                prompt = translated_prompt + "\n\n" + cleaned_text
            else:
                prompt = PODCAST_PROMPT + "\n\n" + cleaned_text
            logger.debug(f"LLM Prompt for Episode {chapter_number}:\n{prompt[:500]}...")
            logger.debug(f"Using language: {language}")
            logger.debug(f"Sample of cleaned prompt text:\n{cleaned_text[:500]}...")
            podcast_script = self.llm_client.generate_podcast_script(
                prompt,
                language=self.resolved_language if self.translate else self.translate,
            )

        # Save the script if requested
        if self.save_text:
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(f"# Podcast Episode {chapter_number}\n")
                    f.write(f"# Generated from: {self.title}\n\n")
                    f.write(podcast_script)
                with open(
                    self.scripts_path / f"original_text_{chapter_number:03d}.txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(chapter_text)
                logger.info(f"Saved podcast script to: {script_path}")
            except IOError as e:
                logger.error(f"Failed to save script for Episode {chapter_number}: {e}")

        return podcast_script

    def synthesize_episode(self, podcast_script: str, episode_number: int) -> Path:
        """Synthesizes a single podcast episode from script."""
        logger.info(f"Synthesizing Podcast Episode {episode_number}...")
        episode_wav_path = self.episodes_path / f"episode_{episode_number:03d}.wav"
        episode_mp3_path = episode_wav_path.with_suffix(".mp3")

        if episode_mp3_path.exists():
            logger.info(
                f"Episode {episode_number} MP3 already exists, skipping synthesis."
            )
            return episode_mp3_path

        if not podcast_script.strip():
            logger.warning(f"Episode {episode_number} contains no text to synthesize.")
            return episode_mp3_path

        # Break script into sentences for better TTS processing
        sentences = self._break_script_into_segments(podcast_script)

        if not sentences:
            logger.warning(
                f"No sentences extracted from Episode {episode_number} script."
            )
            return episode_mp3_path

        # Translate if needed
        if self.translate and self.language:
            logger.info(
                f"Translating {len(sentences)} segments for Episode {episode_number}..."
            )
            try:
                from audify.translate import translate_sentence

                sentences = [
                    translate_sentence(
                        sentence, src_lang=self.language, tgt_lang=self.translate
                    )
                    for sentence in tqdm.tqdm(
                        sentences,
                        desc=f"Translating Ep. {episode_number}",
                        unit="segment",
                    )
                ]
            except Exception as e:
                logger.error(
                    f"Error translating episode {episode_number}: {e}",
                    exc_info=True,
                )
                logger.warning("Proceeding with original text for synthesis.")

        # Synthesize audio
        self._synthesize_sentences(sentences, episode_wav_path)
        return self._convert_to_mp3(episode_wav_path)

    def _break_script_into_segments(self, script: str) -> List[str]:
        """Break podcast script into segments suitable for TTS."""
        from audify.utils import break_text_into_sentences

        # First break into sentences
        sentences = break_text_into_sentences(script)

        # For podcast content, we might want longer segments
        # Combine short sentences into longer segments
        segments = []
        current_segment = ""

        for sentence in sentences:
            # If adding this sentence would make segment too long,
            # finalize current segment
            if current_segment and len(current_segment + " " + sentence) > 200:
                segments.append(current_segment.strip())
                current_segment = sentence
            else:
                if current_segment:
                    current_segment += " " + sentence
                else:
                    current_segment = sentence

        # Add the last segment
        if current_segment.strip():
            segments.append(current_segment.strip())

        return [seg for seg in segments if seg.strip()]

    def create_podcast_series(self) -> List[Path]:
        """Create podcast series from all chapters."""
        logger.info("Starting podcast series creation...")

        # Get chapters based on file type
        if isinstance(self.reader, EpubReader):
            chapters = self.reader.get_chapters()
        else:
            # For PDF, treat the whole document as one episode
            chapters = [self.reader.cleaned_text]

        num_chapters = len(chapters)
        if self.max_chapters:
            num_chapters = min(num_chapters, self.max_chapters)
            chapters = chapters[:num_chapters]

        logger.info(f"Creating podcast series with {num_chapters} episodes...")

        if self.confirm:
            response = input(f"Create {num_chapters} podcast episodes? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                logger.info("Podcast creation cancelled by user.")
                return []

        episode_paths = []

        for i, chapter_content in enumerate(
            tqdm.tqdm(chapters, desc="Creating Podcast Episodes", unit="episode")
        ):
            episode_number = i + 1

            try:
                # Generate podcast script
                podcast_script = self.generate_podcast_script(
                    chapter_content,
                    episode_number,
                    language=self.resolved_language
                    if self.translate
                    else self.translate,
                )
                logger.debug(
                    f"Podcast Script for Episode {episode_number}:"
                    f"\n{podcast_script[:500]}..."
                )

                # Synthesize episode
                episode_path = self.synthesize_episode(podcast_script, episode_number)

                if episode_path.exists():
                    episode_paths.append(episode_path)
                    logger.info(
                        f"Successfully created Episode {episode_number}: {episode_path}"
                    )
                else:
                    logger.warning(f"Failed to create Episode {episode_number}")

            except Exception as e:
                logger.error(
                    f"Error creating Episode {episode_number}: {e}",
                    exc_info=True,
                )
                continue

        logger.info(
            f"Podcast series creation complete. Created {len(episode_paths)} episodes."
        )

        # Create M4B audiobook from episodes if we have episodes
        if episode_paths:
            self.create_m4b()

        return episode_paths

    def _initialize_metadata_file(self) -> None:
        """Writes the initial header to the FFmpeg metadata file."""
        self.metadata_path = self.podcast_path / "chapters.txt"
        logger.info(f"Initializing metadata file: {self.metadata_path}")
        try:
            with open(self.metadata_path, "w") as f:
                f.write(";FFMETADATA1\n")
                f.write("major_brand=M4A\n")
                f.write("minor_version=512\n")
                f.write("compatible_brands=M4A isis2\n")
                f.write("encoder=Lavf61.7.100\n")
        except IOError as e:
            logger.error(f"Failed to initialize metadata file: {e}", exc_info=True)
            raise

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

    def _log_episode_metadata(
        self,
        episode_number: int,
        start_time_ms: int,
        duration_s: float,
        chapter_title: Optional[str] = None,
    ) -> int:
        """Appends episode metadata to the FFmpeg metadata file."""
        end_time_ms = start_time_ms + int(duration_s * 1000)
        title = chapter_title or f"Episode {episode_number}"

        logger.debug(
            f"Logging metadata for '{title}': Start={start_time_ms}, "
            f"End={end_time_ms}, Duration={duration_s:.2f}s"
        )
        try:
            with open(self.metadata_path, "a") as f:
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={start_time_ms}\n")
                f.write(f"END={end_time_ms}\n")
                f.write(f"title={title}\n")
            return end_time_ms
        except IOError as e:
            logger.error(
                f"Failed to write episode metadata for '{title}': {e}", exc_info=True
            )
            return start_time_ms

    def _build_ffmpeg_command(
        self,
    ) -> Tuple[List[str], Optional[tempfile._TemporaryFileWrapper]]:
        """Builds the FFmpeg command for M4B creation."""
        cover_args = []
        cover_temp_file = None

        if (
            hasattr(self, "cover_image_path")
            and isinstance(self.cover_image_path, Path)
            and self.cover_image_path.exists()
        ):
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
            cover_args = ["-map", "0:a"]

        command = [
            "ffmpeg",
            "-i",
            str(self.temp_m4b_path),
            "-i",
            str(self.metadata_path),
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

    def create_m4b(self) -> None:
        """Combines podcast episodes into M4B audiobook file."""
        logger.info("Starting M4B creation process for podcast...")

        # Get all episode MP3 files
        episode_mp3_files = sorted(self.episodes_path.glob("episode_*.mp3"))

        if not episode_mp3_files:
            logger.error("No episode MP3 files found to create M4B.")
            return

        # Set up paths for M4B creation
        self.temp_m4b_path = self.podcast_path / f"{self.file_name}.tmp.m4b"
        self.final_m4b_path = self.podcast_path / f"{self.file_name}.m4b"

        # Initialize metadata file
        self._initialize_metadata_file()

        # Calculate total duration
        total_duration_hours = self._calculate_total_duration(episode_mp3_files) / 3600
        logger.info(f"Total podcast duration: {total_duration_hours:.2f} hours")

        # Create single M4B (podcasts are typically shorter than 15 hours)
        logger.info("Creating M4B audiobook from podcast episodes...")
        self._create_single_m4b(episode_mp3_files)

    def _create_single_m4b(self, episode_mp3_files: List[Path]) -> None:
        """Create a single M4B file from all podcast episodes."""
        if not self.temp_m4b_path.exists():
            logger.info(f"Combining {len(episode_mp3_files)} episode MP3s...")
            combined_audio = AudioSegment.empty()
            current_start_time_ms = 0

            for i, mp3_file in enumerate(
                tqdm.tqdm(episode_mp3_files, desc="Combining Episodes", unit="file")
            ):
                try:
                    audio = AudioSegment.from_mp3(mp3_file)
                    combined_audio += audio
                    # Log metadata for this episode
                    duration_s = len(audio) / 1000.0
                    episode_number = i + 1
                    current_start_time_ms = self._log_episode_metadata(
                        episode_number,
                        current_start_time_ms,
                        duration_s,
                        chapter_title=self.chapter_titles[i]
                        if i < len(self.chapter_titles)
                        else None,
                    )

                except CouldntDecodeError:
                    logger.error(
                        f"Could not decode episode file: {mp3_file}, skipping."
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing episode file {mp3_file}: {e}, skipping.",
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
                f"Temporary M4B file already exists: {self.temp_m4b_path}. "
                "Skipping combination."
            )

        # Add metadata and cover image using FFmpeg
        logger.info("Adding metadata and cover image using FFmpeg...")
        ffmpeg_command, cover_temp_file = self._build_ffmpeg_command()

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

            logger.info(f"M4B audiobook created: {self.final_m4b_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed with exit code {e.returncode}")
            logger.error(f"FFmpeg stdout:\n{e.stdout}")
            logger.error(f"FFmpeg stderr:\n{e.stderr}")
            logger.error(
                "M4B creation failed. The temporary M4B file and episode files are "
                "preserved for inspection."
            )
            raise
        except FileNotFoundError:
            logger.error(
                "FFmpeg command not found. Please ensure FFmpeg is installed and in "
                "your system's PATH."
            )
            raise
        finally:
            if cover_temp_file:
                try:
                    cover_temp_file.close()
                    Path(cover_temp_file.name).unlink(missing_ok=True)
                    logger.debug(
                        f"Cleaned up temporary cover file:{cover_temp_file.name}"
                    )
                except Exception as e_clean:
                    logger.warning(
                        f"Error cleaning up temporary cover file"
                        f"{cover_temp_file.name}: {e_clean}"
                    )

    def synthesize(self) -> Path:
        """Main synthesis method - creates the podcast series."""
        logger.info(f"Starting podcast creation for: {self.path.name}")
        episode_paths = self.create_podcast_series()

        if episode_paths:
            logger.info(f"Podcast series complete with {len(episode_paths)} episodes")
            return self.podcast_path
        else:
            logger.error("No podcast episodes were created successfully")
            return self.podcast_path


class PodcastEpubCreator(PodcastCreator):
    """Specialized podcast creator for EPUB files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.reader, "get_chapters"):
            raise ValueError("PodcastEpubCreator requires an EPUB reader")


class PodcastPdfCreator(PodcastCreator):
    """Specialized podcast creator for PDF files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.reader, "get_cleaned_text"):
            raise ValueError("PodcastPdfCreator requires a PDF reader")

    def create_podcast_series(self) -> List[Path]:
        """Create a single podcast episode from the PDF content."""
        logger.info("Creating podcast episode from PDF...")

        # Get the full PDF content
        pdf_text = self.reader.get_cleaned_text()  # type: ignore # TODO: fix typing
        logger.info(f"Extracted PDF text length: {len(pdf_text.split())} words")
        if not pdf_text.strip():
            logger.error("No text found in PDF")
            return []

        if self.confirm:
            response = input("Create podcast episode from PDF? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                logger.info("Podcast creation cancelled by user.")
                return []

        try:
            logger.info(f"Using language: {self.language}")
            if len(pdf_text.split()) < 200:
                logger.warning(
                    "PDF has very little text. The generated podcast may be very short."
                )
            logger.debug(f"Sample of cleaned prompt text:\n{pdf_text[:500]}...")
            if not self.language == "en":
                translated_prompt = translate_sentence(
                    PODCAST_PROMPT, src_lang="en", tgt_lang=self.language
                )
                prompt = translated_prompt + "\n\n" + pdf_text
            else:
                prompt = PODCAST_PROMPT + "\n\n" + pdf_text
            logger.info(f"LLM Prompt for PDF Episode:\n{prompt[:500]}...")
            # Generate podcast script
            podcast_script = self.generate_podcast_script(
                prompt,
                1,
                self.resolved_language if self.translate else self.translate,
            )

            logger.info(f"Podcast Script for PDF Episode:\n{podcast_script[:500]}...")
            # Synthesize episode
            episode_path = self.synthesize_episode(podcast_script, 1)

            if episode_path.exists():
                logger.info(f"Successfully created podcast episode: {episode_path}")
                episode_paths = [episode_path]
                return episode_paths
            else:
                logger.warning("Failed to create podcast episode")
                return []

        except Exception as e:
            logger.error(f"Error creating podcast episode: {e}", exc_info=True)
            return []
