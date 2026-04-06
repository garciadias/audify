import re
from pathlib import Path
from typing import Any, List, Optional, Union

import tqdm
from bs4 import BeautifulSoup
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from audify.readers.ebook import EpubReader
from audify.readers.pdf import PdfReader
from audify.text_to_speech import BaseSynthesizer
from audify.translate import translate_sentence
from audify.utils.api_config import CommercialAPIConfig, OllamaAPIConfig
from audify.utils.audio import AudioProcessor
from audify.utils.constants import (
    DEFAULT_TTS_PROVIDER,
    OLLAMA_API_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OUTPUT_BASE_DIR,
)
from audify.utils.logging_utils import setup_logging
from audify.utils.m4b_builder import (
    append_chapter_metadata,
    assemble_m4b,
    write_metadata_header,
)
from audify.utils.prompts import AUDIOBOOK_PROMPT
from audify.utils.text import break_text_into_sentences, clean_text, get_file_name_title

# Default LLM parameters for audiobook generation
_DEFAULT_LLM_PARAMS = {
    "num_ctx": 8 * 4096,
    "temperature": 0.8,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "seed": 428798,
    "top_k": 60,
    "num_predict": 4096,
}

# Configure logging
logger = setup_logging(module_name=__name__)


class LLMClient:
    """Client for interacting with LLM services (Ollama or commercial APIs)."""

    def __init__(
        self, base_url: str = OLLAMA_API_BASE_URL, model: str = OLLAMA_DEFAULT_MODEL
    ):
        """Initialize LLM client.

        Args:
            base_url: Base URL for Ollama API (ignored for commercial APIs)
            model: Model name. Use 'api:model_name' for commercial APIs
                (e.g., 'api:deepseek/deepseek-chat')
        """
        self.model_string = model  # Keep original for passing to translate
        self.base_url = base_url
        # Check if using commercial API (format: api:model_name)
        if model.startswith("api:"):
            self.is_commercial = True
            actual_model = model[4:]  # Remove 'api:' prefix
            self.config: Union[OllamaAPIConfig, CommercialAPIConfig] = (
                CommercialAPIConfig(model=actual_model)
            )
            logger.info(f"Using commercial API with model: {actual_model}")
        else:
            self.is_commercial = False
            self.config = OllamaAPIConfig(base_url=base_url, model=model)
            logger.info(f"Using Ollama with model: {model}")

    def generate_script(
        self,
        text: str,
        prompt: str,
        language: Optional[str] = None,
        **llm_params,
    ) -> str:
        """Generate a script from text using a custom prompt and LLM.

        Args:
            text: The source text to transform.
            prompt: The system prompt/instructions for the LLM.
            language: Target language code. If not "en", prompt is translated.
            **llm_params: Override default LLM parameters (temperature, top_p, etc.).

        Returns:
            The generated script text, or an error message on failure.
        """
        # Merge default params with overrides
        params: dict[str, Any] = dict(_DEFAULT_LLM_PARAMS)
        params.update(llm_params)

        # Prepare system prompt (translate if needed)
        if language and language != "en":
            system_prompt = translate_sentence(
                prompt,
                model=self.model_string,
                src_lang="en",
                tgt_lang=language,
                base_url=self.base_url,
            )
        else:
            system_prompt = prompt

        try:
            if self.is_commercial:
                logger.info(f"Sending request to commercial API: {self.config.model}")
            else:
                logger.info(f"Sending request to LLM at {self.config.base_url}")

            response = self.config.generate(
                system_prompt=system_prompt,
                user_prompt=text,
                **params,
            )

            # Clean the response
            if response:
                # if it is a reasoning model, eliminate the reasoning steps
                if "think" in response.lower():
                    return clean_text(response.split("</think>")[-1].strip())
                return clean_text(response.strip())
            else:
                logger.error("Empty response from LLM")
                return "Error: Unable to generate script for this content."

        except Exception as e:
            logger.error(f"Error communicating with LLM: {e}")
            if "connection" in str(e).lower():
                if self.is_commercial:
                    return (
                        "Error: Could not connect to commercial API. "
                        "Please check your API key and internet connection."
                    )
                else:
                    return (
                        "Error: Could not connect to local LLM server. "
                        f"Please ensure Ollama is running at {self.config.base_url}."
                    )
            elif "timeout" in str(e).lower():
                return "Error: Request to LLM timed out. Content might be too long."
            elif "api" in str(e).lower() and "key" in str(e).lower():
                return (
                    "Error: API key issue. Please ensure your API key is "
                    "properly configured in the .keys file or environment "
                    "variables."
                )
            else:
                return f"Error: Failed to generate script due to: {str(e)}"

    def generate_audiobook_script(
        self, chapter_text: str, language: Optional[str]
    ) -> str:
        """Generate audiobook script from chapter text using LLM.

        Backward-compatible wrapper around generate_script() using
        the default AUDIOBOOK_PROMPT.
        """
        return self.generate_script(
            text=chapter_text, prompt=AUDIOBOOK_PROMPT, language=language
        )


class AudiobookCreator(BaseSynthesizer):
    """Creates audiobooks from ebook content using LLM and TTS."""

    def __init__(
        self,
        path: str | Path,
        language: Optional[str] = None,
        voice: str = "af_bella",
        model_name: str = OLLAMA_DEFAULT_MODEL,
        translate: Optional[str] = None,
        save_text: bool = True,  # Default to True for audiobook scripts
        llm_base_url: str = OLLAMA_API_BASE_URL,
        llm_model: str = OLLAMA_DEFAULT_MODEL,
        max_chapters: Optional[int] = None,
        confirm: bool = True,
        output_dir: Optional[str | Path] = None,
        tts_provider: Optional[str] = None,
        llm_config: Optional[Union[OllamaAPIConfig, CommercialAPIConfig]] = None,
        task: Optional[str] = None,
        prompt_file: Optional[str | Path] = None,
    ):
        # Initialize file reader based on extension
        self.reader: Union[EpubReader, PdfReader]
        file_path = Path(path)
        # Build llm_config for the reader from the user's model if not provided
        if llm_config is None and llm_model:
            if llm_model.startswith("api:"):
                llm_config = CommercialAPIConfig(model=llm_model[4:])
            else:
                llm_config = OllamaAPIConfig(base_url=llm_base_url, model=llm_model)
        if file_path.suffix.lower() == ".epub":
            self.reader = EpubReader(path, llm_config=llm_config)
            detected_language = self.reader.get_language()
            self.title = self.reader.title
        elif file_path.suffix.lower() == ".pdf":
            self.reader = PdfReader(path)
            detected_language = "en"  # PDF reader will not detect language on metadata
            self.title = file_path.stem
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        self.language = language if language else detected_language
        # Keep resolved_language for callers/tests that expect this attribute
        self.resolved_language = self.language
        if not self.language:
            raise ValueError(
                "Language must be provided or detectable from the file metadata."
            )

        # Setup output paths
        self.output_base_dir = Path(output_dir or OUTPUT_BASE_DIR).resolve()
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Create audiobook-specific title
        self.audiobook_title = f"Audiobook - {self.title}"
        # Use file_name as the audiobook file_name
        # Clean file name to remove invalid characters
        self.file_name = Path(get_file_name_title(file_path.stem))
        self._setup_paths(self.file_name)

        # Initialize LLM client
        self.llm_client = LLMClient(llm_base_url, llm_model)
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.max_chapters = max_chapters
        self.confirm = confirm

        # Resolve task prompt and LLM parameters
        self.task_name = task or "audiobook"
        self._prompt_file = prompt_file
        self._requires_llm = True  # Default, will be updated in _resolve_task_prompt
        self._resolve_task_prompt()

        # Initialize parent class
        super().__init__(
            path=path,
            language=self.language,
            voice=voice,
            model_name=model_name,
            translate=translate,
            save_text=save_text,
            tts_provider=tts_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )

        # Setup cover image if available
        if isinstance(self.reader, EpubReader):
            self.cover_image_path: Optional[Path] = self.reader.get_cover_image(
                self.audiobook_path
            )
        else:
            self.cover_image_path = None
        self.chapter_titles: List[str] = []

    def _resolve_task_prompt(self) -> None:
        """Resolve the prompt and LLM params from task name or prompt file."""
        from audify.prompts.manager import PromptManager
        from audify.prompts.tasks import TaskRegistry

        manager = PromptManager()

        if self._prompt_file:
            self._task_prompt = manager.load_prompt_file(self._prompt_file)
            self._task_llm_params: dict = dict(_DEFAULT_LLM_PARAMS)
            self._requires_llm = True  # Custom prompt implies LLM
            logger.info(f"Using custom prompt from: {self._prompt_file}")
        else:
            task_config = TaskRegistry.get(self.task_name)
            if task_config:
                self._task_prompt = task_config.prompt
                self._task_llm_params = task_config.get_llm_params()
                self._requires_llm = task_config.requires_llm
                logger.info(f"Using task '{self.task_name}' prompt")
            else:
                # Fallback: try loading as a builtin prompt
                try:
                    self._task_prompt = manager.get_builtin_prompt(self.task_name)
                    self._task_llm_params = dict(_DEFAULT_LLM_PARAMS)
                    # Assume requires LLM for builtin prompts except "direct"
                    self._requires_llm = self.task_name != "direct"
                except FileNotFoundError:
                    # Default to audiobook prompt
                    self._task_prompt = AUDIOBOOK_PROMPT
                    self._task_llm_params = dict(_DEFAULT_LLM_PARAMS)
                    self._requires_llm = True
                    logger.warning(
                        f"Unknown task '{self.task_name}', "
                        "falling back to audiobook prompt"
                    )

    def _setup_paths(self, file_name_base: Path) -> None:
        """Sets up the necessary output paths for audiobook creation."""
        if not self.output_base_dir.exists():
            logger.info(f"Creating output base directory: {self.output_base_dir}")
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

        folder_safe_name = re.sub(r"[^\w\s-]", "", file_name_base.stem).strip()
        self.audiobook_path = self.output_base_dir / folder_safe_name
        self.audiobook_path.mkdir(parents=True, exist_ok=True)
        self.scripts_path = self.audiobook_path / "scripts"
        self.scripts_path.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.audiobook_path / "episodes"
        self.episodes_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Audiobook output directory set to: {self.audiobook_path}")

    def _clean_text_for_audiobook(self, text: str) -> str:
        """
        Clean text by removing references, citations, and other non-content elements.
        """
        text = str(text)  # Ensure text is a string
        # If there are html tags, use bs4 to extract text
        if re.search(r"<[^>]+>", text):
            text = BeautifulSoup(text, "html.parser").get_text()
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

    def generate_audiobook_script(
        self, chapter_text: str, chapter_number: int, language: Optional[str] = None
    ) -> str:
        """Generate audiobook script for a single chapter."""

        logger.info(f"Generating audiobook script for Chapter {chapter_number}...")

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
        cleaned_text = self._clean_text_for_audiobook(chapter_text)
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

        # Determine effective language.
        # Prefer explicit param, then translate, then file language
        effective_language = (
            language
            if language
            else (self.translate if self.translate else self.language)
        )

        # If an explicit translate target was requested,
        # translate the cleaned text
        if self.translate:
            # prefer explicit language attrs; fall back to resolved_language
            src_lang = getattr(self, "language", None) or getattr(
                self, "resolved_language", None
            )
            cleaned_text = translate_sentence(
                cleaned_text,
                model=self.llm_model,
                src_lang=src_lang,
                tgt_lang=self.translate,
                base_url=self.llm_base_url,
            )

        # Generate audiobook script using LLM or direct text
        if getattr(self, "_requires_llm", True) is False:
            task_name = getattr(self, "task_name", "unknown")
            logger.info(
                f"Skipping LLM for task '{task_name}', using cleaned text directly"
            )
            audiobook_script = cleaned_text
        elif len(cleaned_text.split()) < 200:
            logger.warning(
                f"Chapter {chapter_number} has very little text after cleaning. "
                "The generated audiobook may be very short."
            )
            audiobook_script = cleaned_text
        else:
            logger.debug(f"Using language: {effective_language}")
            logger.debug(f"Sample of chapter text:\n{cleaned_text[:500]}...")

            audiobook_script = self.llm_client.generate_script(
                text=cleaned_text,
                prompt=self._task_prompt,
                language=effective_language,
                **self._task_llm_params,
            )

        # Save the script if requested
        if self.save_text:
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(audiobook_script)
                with open(
                    self.scripts_path / f"original_text_{chapter_number:03d}.txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(cleaned_text)
                logger.info(f"Saved audiobook script to: {script_path}")
            except IOError as e:
                logger.error(f"Failed to save script for Episode {chapter_number}: {e}")

        return audiobook_script

    def synthesize_episode(self, audiobook_script: str, episode_number: int) -> Path:
        """Synthesizes a single audiobook episode from script."""
        logger.info(f"Synthesizing Audiobook Episode {episode_number}...")
        # Use standardized episode file naming (tests expect episode_###.wav)
        episode_wav_path = self.episodes_path / f"episode_{episode_number:03d}.wav"
        episode_mp3_path = episode_wav_path.with_suffix(".mp3")

        if episode_mp3_path.exists():
            logger.info(
                f"Episode {episode_number} MP3 already exists, skipping synthesis."
            )
            return episode_mp3_path

        if not audiobook_script.strip():
            logger.warning(f"Episode {episode_number} contains no text to synthesize.")
            return episode_mp3_path

        # Break script into sentences for better TTS processing
        sentences = self._break_script_into_segments(audiobook_script)

        # If translation is enabled, translate each sentence before synthesis
        if getattr(self, "translate", None):
            translated_sentences = []
            src_lang = getattr(self, "language", None) or getattr(
                self, "resolved_language", None
            )
            for s in sentences:
                try:
                    translated_sentences.append(
                        translate_sentence(
                            s,
                            model=self.llm_model,
                            src_lang=src_lang,
                            tgt_lang=self.translate,
                            base_url=self.llm_base_url,
                        )
                    )
                except Exception:
                    # On translation failure, fall back to original sentence
                    translated_sentences.append(s)
            sentences = translated_sentences

        if not sentences:
            logger.warning(
                f"No sentences extracted from Episode {episode_number} script."
            )
            return episode_mp3_path

        # Synthesize audio
        self._synthesize_sentences(sentences, episode_wav_path)
        return self._convert_to_mp3(episode_wav_path)

    def _break_script_into_segments(self, script: str) -> List[str]:
        """Break audiobook script into segments suitable for TTS."""

        # First break into sentences
        sentences = break_text_into_sentences(script)

        # For audiobook content, we might want longer segments
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

    def create_audiobook_series(self) -> List[Path]:
        """Create audiobook series from all chapters."""
        logger.info("Starting audiobook series creation...")

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

        logger.info(f"Creating audiobook series with {num_chapters} episodes...")

        if self.confirm:
            response = input(f"Create {num_chapters} audiobook episodes? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                logger.info("Audiobook creation cancelled by user.")
                return []

        episode_paths = []

        for i, chapter_content in enumerate(
            tqdm.tqdm(chapters, desc="Creating Audiobook Episodes", unit="episode")
        ):
            episode_number = i + 1

            try:
                # Generate audiobook script
                audiobook_script = self.generate_audiobook_script(
                    chapter_content,
                    episode_number,
                )
                logger.debug(
                    f"Audiobook Script for Episode {episode_number}:"
                    f"\n{audiobook_script[:500]}..."
                )

                # Synthesize episode
                episode_path = self.synthesize_episode(audiobook_script, episode_number)

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
            "Audiobook series creation complete. "
            f"Created {len(episode_paths)} episodes."
        )

        # Create M4B audiobook from episodes if we have episodes
        if episode_paths:
            self.create_m4b()

        return episode_paths

    def _initialize_metadata_file(self) -> None:
        """Writes the initial header to the FFmpeg metadata file."""
        self.metadata_path = self.audiobook_path / "chapters.txt"
        write_metadata_header(self.metadata_path)

    def _calculate_total_duration(self, mp3_files: List[Path]) -> float:
        """Calculate total duration of MP3 files in seconds."""
        total_duration = 0.0
        for mp3_file in mp3_files:
            try:
                duration = AudioProcessor.get_duration(str(mp3_file))
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
        title = chapter_title or f"Episode {episode_number}"
        return append_chapter_metadata(
            self.metadata_path, title, start_time_ms, duration_s
        )

    def create_m4b(self) -> None:
        """Combines audiobook episodes into M4B audiobook file."""
        logger.info("Starting M4B creation process for audiobook...")

        episode_mp3_files = sorted(self.episodes_path.glob("episode_*.mp3"))
        if not episode_mp3_files:
            logger.error("No episode MP3 files found to create M4B.")
            return

        self.temp_m4b_path = self.audiobook_path / f"{self.file_name}.tmp.m4b"
        self.final_m4b_path = self.audiobook_path / f"{self.file_name}.m4b"

        self._initialize_metadata_file()

        total_duration_hours = self._calculate_total_duration(episode_mp3_files) / 3600
        logger.info(f"Total audiobook duration: {total_duration_hours:.2f} hours")

        logger.info("Creating M4B audiobook from audiobook episodes...")
        self._create_single_m4b(episode_mp3_files)

    def _create_single_m4b(self, episode_mp3_files: List[Path]) -> None:
        """Create a single M4B file from all audiobook episodes."""
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
                    duration_s = len(audio) / 1000.0
                    current_start_time_ms = self._log_episode_metadata(
                        i + 1,
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
                f"Temporary M4B already exists: {self.temp_m4b_path}. Skipping."
            )

        logger.info("Adding metadata and cover image using FFmpeg...")
        assemble_m4b(
            self.temp_m4b_path,
            self.metadata_path,
            self.final_m4b_path,
            getattr(self, "cover_image_path", None),
        )
        logger.info(f"M4B audiobook created: {self.final_m4b_path}")

    def synthesize(self) -> Path:
        """Main synthesis method - creates the audiobook series."""
        logger.info(f"Starting audiobook creation for: {self.path.name}")
        episode_paths = self.create_audiobook_series()

        if episode_paths:
            logger.info(f"Audiobook series complete with {len(episode_paths)} episodes")
            return self.audiobook_path
        else:
            logger.error("No audiobook episodes were created successfully")
            return self.audiobook_path


class AudiobookEpubCreator(AudiobookCreator):
    """Specialized audiobook creator for EPUB files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.reader, "get_chapters"):
            raise ValueError("AudiobookEpubCreator requires an EPUB reader")


class AudiobookPdfCreator(AudiobookCreator):
    """Specialized audiobook creator for PDF files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_audiobook_series(self) -> List[Path]:
        """Create a single audiobook episode from the PDF content."""
        logger.info("Creating audiobook episode from PDF...")

        # Get the full PDF content
        if isinstance(self.reader, PdfReader):
            pdf_text = self.reader.cleaned_text
        else:
            raise ValueError("AudiobookPdfCreator requires a PDF reader")
        logger.info(f"Extracted PDF text length: {len(pdf_text.split())} words")
        if not pdf_text.strip():
            logger.error("No text found in PDF")
            return []

        if self.confirm:
            response = input("Create audiobook episode from PDF? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                logger.info("Audiobook creation cancelled by user.")
                return []

        try:
            logger.info(f"Using language: {self.language}")
            if len(pdf_text.split()) < 200:
                logger.warning(
                    "PDF has very little text. "
                    "The generated audiobook may be very short."
                )

            # Generate audiobook script (now uses system/user role pattern)
            audiobook_script = self.generate_audiobook_script(
                pdf_text,
                1,
            )

            # Synthesize episode
            episode_path = self.synthesize_episode(audiobook_script, 1)

            if episode_path.exists():
                logger.info(f"Successfully created audiobook episode: {episode_path}")
                episode_paths = [episode_path]
                return episode_paths
            else:
                logger.warning("Failed to create audiobook episode")
                return []

        except Exception as e:
            logger.error(f"Error creating audiobook episode: {e}", exc_info=True)
            return []


class DirectoryAudiobookCreator:
    """Creates a single audiobook from multiple files in a directory."""

    def __init__(
        self,
        directory_path: str | Path,
        language: str = "en",
        voice: str = "af_bella",
        model_name: str = "kokoro",
        translate: Optional[str] = None,
        save_text: bool = True,
        llm_base_url: str = OLLAMA_API_BASE_URL,
        llm_model: str = OLLAMA_DEFAULT_MODEL,
        confirm: bool = True,
        output_dir: Optional[str | Path] = None,
        tts_provider: Optional[str] = None,
        task: Optional[str] = None,
        prompt_file: Optional[str | Path] = None,
    ):
        self.directory_path = Path(directory_path)
        if not self.directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        self.language = language
        self.voice = voice
        self.model_name = model_name
        self.translate = translate
        self.save_text = save_text
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.confirm = confirm
        self.tts_provider = tts_provider or DEFAULT_TTS_PROVIDER
        self.task = task
        self.prompt_file = prompt_file

        # Setup output paths
        self.output_base_dir = Path(output_dir or OUTPUT_BASE_DIR).resolve()
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Create directory-specific title
        self.title = self.directory_path.name
        self.file_name = Path(get_file_name_title(self.title))
        self._setup_paths(self.file_name)

        # Initialize TTS synthesizer for titles
        from audify.text_to_speech import BaseSynthesizer

        self.tts_synthesizer = BaseSynthesizer(
            path=str(self.directory_path),
            language=self.language,
            voice=self.voice,
            model_name=self.model_name,
            translate=self.translate,
            save_text=False,
            tts_provider=self.tts_provider,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
        )

        self.chapter_titles: List[str] = []
        self.episode_paths: List[Path] = []

    def _setup_paths(self, file_name_base: Path) -> None:
        """Sets up the necessary output paths for audiobook creation."""
        if not self.output_base_dir.exists():
            logger.info(f"Creating output base directory: {self.output_base_dir}")
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

        folder_safe_name = re.sub(r"[^\w\s-]", "", file_name_base.stem).strip()
        self.audiobook_path = self.output_base_dir / folder_safe_name
        self.audiobook_path.mkdir(parents=True, exist_ok=True)
        self.scripts_path = self.audiobook_path / "scripts"
        self.scripts_path.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.audiobook_path / "episodes"
        self.episodes_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Audiobook output directory set to: {self.audiobook_path}")

    def _get_supported_files(self) -> List[Path]:
        """Get all supported files from the directory."""
        supported_extensions = {".epub", ".pdf", ".txt", ".md"}
        files = []
        for file_path in sorted(self.directory_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        logger.info(f"Found {len(files)} supported files in directory")
        return files

    def _synthesize_title_audio(
        self, title: str, episode_number: int
    ) -> Optional[Path]:
        """Synthesize the title as audio."""
        logger.info(f"Synthesizing title for episode {episode_number}: {title}")
        title_wav_path = self.episodes_path / f"title_{episode_number:03d}.wav"
        title_mp3_path = title_wav_path.with_suffix(".mp3")

        if title_mp3_path.exists():
            logger.info(f"Title audio for episode {episode_number} already exists")
            return title_mp3_path

        # Create title text with pause
        title_text = f"{title}."
        sentences = [title_text]

        try:
            self.tts_synthesizer._synthesize_sentences(sentences, title_wav_path)
            return AudioProcessor.convert_wav_to_mp3(title_wav_path)
        except Exception as e:
            logger.error(f"Error synthesizing title for episode {episode_number}: {e}")
            return None

    def _process_single_file(
        self, file_path: Path, episode_number: int
    ) -> Optional[Path]:
        """Process a single file and create an episode with title."""
        logger.info(f"Processing file {episode_number}: {file_path.name}")

        # Get the article title from filename
        article_title = file_path.stem.replace("_", " ").replace("-", " ")
        self.chapter_titles.append(article_title)

        # Create an audiobook creator for this single file
        try:
            file_extension = file_path.suffix.lower()
            creator: Union[AudiobookEpubCreator, AudiobookPdfCreator]

            if file_extension == ".epub":
                creator = AudiobookEpubCreator(
                    path=str(file_path),
                    language=self.language,
                    voice=self.voice,
                    model_name=self.model_name,
                    translate=self.translate,
                    save_text=self.save_text,
                    llm_base_url=self.llm_base_url,
                    llm_model=self.llm_model,
                    max_chapters=None,
                    confirm=False,
                    output_dir=self.audiobook_path / f"temp_{episode_number:03d}",
                    tts_provider=self.tts_provider,
                    task=self.task,
                    prompt_file=self.prompt_file,
                )
            elif file_extension == ".pdf":
                creator = AudiobookPdfCreator(
                    path=str(file_path),
                    language=self.language,
                    voice=self.voice,
                    model_name=self.model_name,
                    translate=self.translate,
                    save_text=self.save_text,
                    llm_base_url=self.llm_base_url,
                    llm_model=self.llm_model,
                    confirm=False,
                    output_dir=self.audiobook_path / f"temp_{episode_number:03d}",
                    tts_provider=self.tts_provider,
                    task=self.task,
                    prompt_file=self.prompt_file,
                )
            elif file_extension in [".txt", ".md"]:
                # For text files, create a simple episode
                return self._process_text_file(file_path, episode_number, article_title)
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return None

            # Generate audiobook for this file
            creator.create_audiobook_series()

            # Find the generated MP3 files
            if file_extension == ".epub":
                temp_episodes = sorted(creator.episodes_path.glob("episode_*.mp3"))
            else:
                temp_episodes = sorted(creator.episodes_path.glob("episode_*.mp3"))

            if not temp_episodes:
                logger.warning(f"No episodes generated for {file_path.name}")
                return None

            # Combine all episodes from this file
            combined_audio = AudioSegment.empty()

            # Add title audio first
            title_audio_path = self._synthesize_title_audio(
                article_title, episode_number
            )
            if title_audio_path and title_audio_path.exists():
                try:
                    title_audio = AudioSegment.from_mp3(title_audio_path)
                    combined_audio += title_audio
                    # Add a short pause after title
                    combined_audio += AudioSegment.silent(duration=1000)
                except Exception as e:
                    logger.error(f"Error adding title audio: {e}")

            # Add all episode audio
            for episode_path in temp_episodes:
                try:
                    episode_audio = AudioSegment.from_mp3(episode_path)
                    combined_audio += episode_audio
                except Exception as e:
                    logger.error(f"Error adding episode audio from {episode_path}: {e}")

            # Export combined episode
            episode_filename = f"episode_{episode_number:03d}.mp3"
            final_episode_path = self.episodes_path / episode_filename
            combined_audio.export(final_episode_path, format="mp3", bitrate="128k")

            logger.info(f"Created combined episode: {final_episode_path}")
            return final_episode_path

        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}", exc_info=True)
            return None

    def _process_text_file(
        self, file_path: Path, episode_number: int, article_title: str
    ) -> Optional[Path]:
        """Process a text file and create an episode."""
        try:
            # Read the text file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Clean and prepare the text
            cleaned_content = self._clean_text_for_audiobook(content)

            # Generate script using LLM
            llm_client = LLMClient(self.llm_base_url, self.llm_model)

            # Resolve task prompt
            from audify.prompts.manager import PromptManager

            manager = PromptManager()
            prompt = manager.get_prompt(
                task=self.task or "audiobook",
                prompt_file=self.prompt_file,
            )
            audiobook_script = llm_client.generate_script(
                text=cleaned_content,
                prompt=prompt,
                language=self.translate or self.language,
            )

            # Save script if requested
            if self.save_text:
                script_filename = f"episode_{episode_number:03d}_script.txt"
                script_path = self.scripts_path / script_filename
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(audiobook_script)

            # Synthesize title audio
            title_audio_path = self._synthesize_title_audio(
                article_title, episode_number
            )

            # Synthesize content audio
            content_wav_path = self.episodes_path / f"content_{episode_number:03d}.wav"
            sentences = break_text_into_sentences(audiobook_script)

            if self.translate:
                translated_sentences = []
                for s in sentences:
                    try:
                        translated_sentences.append(
                            translate_sentence(
                                s,
                                model=self.llm_model,
                                src_lang=self.language,
                                tgt_lang=self.translate,
                                base_url=self.llm_base_url,
                            )
                        )
                    except Exception:
                        translated_sentences.append(s)
                sentences = translated_sentences

            self.tts_synthesizer._synthesize_sentences(sentences, content_wav_path)
            content_mp3_path = AudioProcessor.convert_wav_to_mp3(content_wav_path)

            # Combine title and content
            combined_audio = AudioSegment.empty()
            if title_audio_path and title_audio_path.exists():
                title_audio = AudioSegment.from_mp3(title_audio_path)
                combined_audio += title_audio
                combined_audio += AudioSegment.silent(duration=1000)

            content_audio = AudioSegment.from_mp3(content_mp3_path)
            combined_audio += content_audio

            # Export final episode
            episode_filename = f"episode_{episode_number:03d}.mp3"
            final_episode_path = self.episodes_path / episode_filename
            combined_audio.export(final_episode_path, format="mp3", bitrate="128k")

            logger.info(f"Created episode from text file: {final_episode_path}")
            return final_episode_path

        except Exception as e:
            logger.error(
                f"Error processing text file {file_path.name}: {e}", exc_info=True
            )
            return None

    def _clean_text_for_audiobook(self, text: str) -> str:
        """Clean text by removing references and non-content elements."""
        text = str(text)
        if re.search(r"<[^>]+>", text):
            text = BeautifulSoup(text, "html.parser").get_text()

        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\([^)]*\d{4}[^)]*\)", "", text)
        text = re.sub(r"\b[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)", "", text)
        text = re.sub(r"doi:\s*[\d\.\w/\-]+", "", text)
        text = re.sub(r"http[s]?://[\w\.\-/\?\=&%]+", "", text)
        text = re.sub(r"www\.[\w\.\-/\?\=&%]+", "", text)
        text = re.sub(r"(?i)\b(figure|fig|table|tab)\s*\.?\s*\d+", "", text)
        text = re.sub(r"\bpp?\.\s*\d+(-\d+)?", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        return text.strip()

    def _initialize_metadata_file(self) -> None:
        """Writes the initial header to the FFmpeg metadata file."""
        self.metadata_path = self.audiobook_path / "chapters.txt"
        write_metadata_header(self.metadata_path)

    def _log_episode_metadata(
        self,
        episode_number: int,
        start_time_ms: int,
        duration_s: float,
        chapter_title: Optional[str] = None,
    ) -> int:
        """Appends episode metadata to the FFmpeg metadata file."""
        title = chapter_title or f"Episode {episode_number}"
        return append_chapter_metadata(
            self.metadata_path, title, start_time_ms, duration_s
        )

    def create_m4b(self) -> None:
        """Combines episodes into M4B audiobook file."""
        logger.info("Starting M4B creation process for directory audiobook...")

        # Get all episode MP3 files
        episode_mp3_files = sorted(self.episodes_path.glob("episode_*.mp3"))

        if not episode_mp3_files:
            logger.error("No episode MP3 files found to create M4B.")
            return

        # Set up paths for M4B creation
        self.temp_m4b_path = self.audiobook_path / f"{self.file_name}.tmp.m4b"
        self.final_m4b_path = self.audiobook_path / f"{self.file_name}.m4b"

        # Initialize metadata file
        self._initialize_metadata_file()

        logger.info(f"Creating M4B audiobook from {len(episode_mp3_files)} episodes...")
        self._create_single_m4b(episode_mp3_files)

    def _create_single_m4b(self, episode_mp3_files: List[Path]) -> None:
        """Create a single M4B file from all episodes."""
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

        logger.info("Adding metadata using FFmpeg...")
        assemble_m4b(
            self.temp_m4b_path,
            self.metadata_path,
            self.final_m4b_path,
            cover_image=None,
        )
        logger.info(f"M4B audiobook created: {self.final_m4b_path}")

    def synthesize(self) -> Path:
        """Main synthesis method - processes all files in directory."""
        logger.info(
            f"Starting directory audiobook creation for: {self.directory_path.name}"
        )

        # Get all supported files
        files = self._get_supported_files()

        if not files:
            logger.error("No supported files found in directory")
            return self.audiobook_path

        if self.confirm:
            prompt = (
                f"Process {len(files)} files from directory "
                f"'{self.directory_path.name}'? (y/N): "
            )
            response = input(prompt)
            if response.lower() not in ["y", "yes"]:
                logger.info("Directory audiobook creation cancelled by user.")
                return self.audiobook_path

        # Process each file
        for i, file_path in enumerate(files, start=1):
            episode_path = self._process_single_file(file_path, i)
            if episode_path and episode_path.exists():
                self.episode_paths.append(episode_path)
                logger.info(
                    f"Successfully processed file {i}/{len(files)}: {file_path.name}"
                )
            else:
                logger.warning(
                    f"Failed to process file {i}/{len(files)}: {file_path.name}"
                )

        if self.episode_paths:
            logger.info(
                f"Directory audiobook processing complete with "
                f"{len(self.episode_paths)} episodes"
            )
            # Create M4B from all episodes
            self.create_m4b()
            return self.audiobook_path
        else:
            logger.error("No episodes were created successfully")
            return self.audiobook_path
