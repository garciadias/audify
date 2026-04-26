import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Union

from bs4 import BeautifulSoup
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from rich.progress import track

from audify.readers.ebook import EpubReader
from audify.readers.pdf import PdfReader
from audify.text_to_speech import BaseSynthesizer, TTSSynthesisError
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
from audify.utils.progress import ProgressIndicator
from audify.utils.prompts import AUDIOBOOK_PROMPT
from audify.utils.text import break_text_into_sentences, clean_text, get_file_name_title

_DEFAULT_LLM_PARAMS = {
    "num_ctx": 8 * 4096,
    "temperature": 0.8,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "seed": 428798,
    "top_k": 60,
    "num_predict": 4096,
}

# Maximum words per chunk sent to the LLM.  At ~4 chars/token and a 4096-token
# output limit, ~3 277 words of audiobook prose fit in one response.  Using
# 2 500 words per input chunk gives the LLM comfortable headroom to expand the
# prose without hitting the output cap.
_MAX_WORDS_PER_LLM_CHUNK = 2500

logger = setup_logging(module_name=__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from environment variables."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clean_text_for_audiobook(text: str) -> str:
    """Remove references, citations, and academic formatting from text."""
    text = str(text)
    if re.search(r"<[^>]+>", text):
        text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\([^)]*\d{4}[^)]*\)", "", text)
    text = re.sub(r"\b[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)", "", text)
    text = re.sub(r"doi:\s*[\d\.\w/\-]+", "", text)
    text = re.sub(r"http[s]?://[\w\.\-/\?\=&%]+", "", text)
    text = re.sub(r"www\.[\w\.\-/\?\=&%]+", "", text)
    for pattern in [
        r"(?i)references?\s*:?\s*\n.*?(?=\n\s*[A-Z]|\Z)",
        r"(?i)bibliography\s*:?\s*\n.*?(?=\n\s*[A-Z]|\Z)",
        r"(?i)works?\s+cited\s*:?\s*\n.*?(?=\n\s*[A-Z]|\Z)",
        r"(?i)literature\s+cited\s*:?\s*\n.*?(?=\n\s*[A-Z]|\Z)",
    ]:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)\b(figure|fig|table|tab)\s*\.?\s*\d+", "", text)
    text = re.sub(r"\bpp?\.\s*\d+(-\d+)?", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


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

        system_prompt = prompt

        try:
            # Prepare system prompt (translate if needed).
            # Cache the result so repeated calls with the same prompt+language
            # (e.g. when processing multiple chunks of a long chapter) only
            # incur one translation request.
            if language and language != "en":
                cache_key = (prompt, language)
                if not hasattr(self, "_prompt_translation_cache"):
                    self._prompt_translation_cache: dict = {}
                if cache_key not in self._prompt_translation_cache:
                    self._prompt_translation_cache[cache_key] = translate_sentence(
                        prompt,
                        model=self.model_string,
                        src_lang="en",
                        tgt_lang=language,
                        base_url=self.base_url,
                    )
                system_prompt = self._prompt_translation_cache[cache_key]
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

    # Class-level default so tests that bypass __init__ still have a valid mode.
    mode: str = "full"

    def __init_progress(self) -> None:
        """Initialize progress indicator if not already done."""
        if not hasattr(self, "_progress"):
            self._progress = ProgressIndicator()

    @property
    def progress(self) -> ProgressIndicator:
        """Lazy-initialized progress indicator."""
        self.__init_progress()
        return self._progress

    @progress.setter
    def progress(self, value: ProgressIndicator) -> None:
        """Set progress indicator."""
        self._progress = value

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
        mode: str = "full",
    ):
        self.reader: Union[EpubReader, PdfReader]
        file_path = Path(path)
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
            detected_language = "en"
            self.title = file_path.stem
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        if mode not in ("full", "process", "synthesize"):
            raise ValueError(
                f"Invalid mode '{mode}': expected 'full', 'process', or 'synthesize'"
            )
        self.mode = mode

        self.language = language if language else detected_language
        self.resolved_language = self.language
        if not self.language:
            raise ValueError(
                "Language must be provided or detectable from the file metadata."
            )

        self.output_base_dir = Path(output_dir or OUTPUT_BASE_DIR).resolve()
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.audiobook_title = f"Audiobook - {self.title}"
        self.file_name = Path(get_file_name_title(file_path.stem))
        self._setup_paths(self.file_name)

        self.llm_client = LLMClient(llm_base_url, llm_model)
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.max_chapters = max_chapters
        self.confirm = confirm
        self.progress = ProgressIndicator()

        # Resolve task prompt and LLM parameters
        self.task_name = task or "audiobook"
        self._prompt_file = prompt_file
        self._requires_llm = True
        self._resolve_task_prompt()

        # For audiobook task, always save scripts to enable resumability
        if self.task_name == "audiobook":
            save_text = True

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
            self._requires_llm = True
            logger.info(f"Using custom prompt from: {self._prompt_file}")
        else:
            task_config = TaskRegistry.get(self.task_name)
            if task_config:
                self._task_prompt = task_config.prompt
                self._task_llm_params = task_config.get_llm_params()
                self._requires_llm = task_config.requires_llm
                logger.info(f"Using task '{self.task_name}' prompt")
            else:
                try:
                    self._task_prompt = manager.get_builtin_prompt(self.task_name)
                    self._task_llm_params = dict(_DEFAULT_LLM_PARAMS)
                    self._requires_llm = self.task_name != "direct"
                except FileNotFoundError:
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

    def _verify_tts_provider_available(self) -> None:
        """Verify TTS provider is available before starting synthesis.

        This check runs early to catch configuration issues before
        wasting time on LLM processing.

        Raises:
            RuntimeError: If TTS provider is not available or misconfigured.
        """
        if _env_flag("AUDIFY_SKIP_TTS_PREFLIGHT", default=False):
            logger.warning(
                "Skipping TTS provider preflight because "
                "AUDIFY_SKIP_TTS_PREFLIGHT is enabled."
            )
            return

        # Some tests build instances via __new__ and don't initialize
        # BaseSynthesizer state. In that case, skip preflight.
        if not hasattr(self, "_tts_config"):
            logger.debug(
                "Skipping TTS preflight because synthesizer state is not initialized."
            )
            return

        logger.info(f"Verifying TTS provider '{self.tts_provider}' availability...")
        tts_config = self._get_tts_config()

        if not tts_config.is_available():
            error_msg = (
                f"TTS provider '{self.tts_provider}' is not available. "
                f"Cannot proceed with audiobook synthesis. "
            )

            # Add provider-specific debugging info
            base_url = getattr(tts_config, "base_url", None)
            if base_url:
                error_msg += f"Configured API URL: {base_url}. "

            error_msg += (
                "Please verify:\n"
                "  1. TTS service is running and accessible\n"
                "  2. Environment variables are correctly set"
                " (TTS provider URL, etc.)\n"
                "  3. Network connectivity to the TTS API"
                " endpoint\n"
                "  4. API credentials if required\n"
            )

            # Add provider-specific guidance
            if self.tts_provider == "kokoro":
                error_msg += (
                    "\nFor Kokoro TTS:\n"
                    "  • Ensure Kokoro service is running\n"
                    "  • Verify KOKORO_API_URL is set correctly\n"
                )

            # For audiobook task, fail fast by default; for other tasks,
            # respect env flag
            strict_preflight = self.task_name == "audiobook" or _env_flag(
                "AUDIFY_STRICT_TTS_PREFLIGHT", default=False
            )
            if strict_preflight:
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.warning(
                error_msg
                + "Continuing anyway (set AUDIFY_STRICT_TTS_PREFLIGHT=1 to fail "
                "fast, or AUDIFY_SKIP_TTS_PREFLIGHT=1 to skip checks entirely)."
            )
            return

        logger.info(f"✓ TTS provider '{self.tts_provider}' is available and ready")

    def _clean_text_for_audiobook(self, text: str) -> str:
        return _clean_text_for_audiobook(text)

    def generate_audiobook_script(
        self, chapter_text: str, chapter_number: int, language: Optional[str] = None
    ) -> str:
        """Generate audiobook script for a single chapter."""

        logger.info(f"Generating audiobook script for Chapter {chapter_number}...")

        # Early exit for empty text (before reader access for safety).
        # Append a placeholder title to keep self.chapter_titles aligned
        # with episode indices used by M4B metadata creation.
        if not chapter_text.strip():
            logger.warning(f"No text found in Chapter {chapter_number}")
            self.chapter_titles.append(f"Chapter {chapter_number}")
            return "This chapter contains no readable text content."

        # Extract chapter title for metadata (needed whether we skip or not)
        chapter_title = (
            self.reader.get_chapter_title(chapter_text)
            if isinstance(self.reader, EpubReader)
            else self.reader.path.stem
        )
        logger.info(f"Chapter {chapter_number} title: {chapter_title}")

        # Check if episode MP3 already exists - if so, skip script generation
        episodes_path = getattr(self, "episodes_path", None)
        if episodes_path is not None:
            episode_mp3_path = episodes_path / f"episode_{chapter_number:03d}.mp3"
            if episode_mp3_path.exists():
                logger.info(
                    f"Episode {chapter_number} MP3 already exists, "
                    "skipping script generation."
                )
                self.chapter_titles.append(chapter_title)
                return ""

        script_path = self.scripts_path / f"episode_{chapter_number:03d}_script.txt"
        if script_path.exists() and not self.confirm:
            logger.info(
                f"Script for Episode {chapter_number} already exists, loading..."
            )
            self.chapter_titles.append(chapter_title)
            logger.info(f"Chapter {chapter_number} title: {chapter_title}")
            with open(script_path, "r", encoding="utf-8") as f:
                return f.read()

        cleaned_text = self._clean_text_for_audiobook(chapter_text)
        logger.info(
            f"Cleaned text for Episode {chapter_number}:"
            " removed references and citations"
        )
        logger.info(
            f"Original length: {len(chapter_text.split())} words, "
            f"Cleaned length: {len(cleaned_text.split())} words"
        )

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
            source_lang = language or getattr(self, "language", "en")
            logger.debug(f"Using language: {source_lang}")
            logger.debug(f"Sample of chapter text:\n{cleaned_text[:500]}...")

            chunks = self._split_text_into_chunks(cleaned_text)
            if len(chunks) == 1:
                audiobook_script = self.llm_client.generate_script(
                    text=cleaned_text,
                    prompt=self._task_prompt,
                    language=source_lang,
                    **self._task_llm_params,
                )
            else:
                logger.info(
                    f"Chapter {chapter_number} ({len(cleaned_text.split())} words) "
                    f"split into {len(chunks)} chunks for LLM processing"
                )
                chunk_scripts: List[str] = []
                for idx, chunk in enumerate(chunks, 1):
                    logger.info(
                        f"  Processing chunk {idx}/{len(chunks)} "
                        f"({len(chunk.split())} words)…"
                    )
                    chunk_script = self.llm_client.generate_script(
                        text=chunk,
                        prompt=self._task_prompt,
                        language=source_lang,
                        **self._task_llm_params,
                    )
                    chunk_scripts.append(chunk_script)
                audiobook_script = " ".join(chunk_scripts)

        self.chapter_titles.append(chapter_title)

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

        try:
            self._synthesize_sentences(sentences, episode_wav_path)
            return self._convert_to_mp3(episode_wav_path)
        except Exception as e:
            logger.error(
                f"TTS synthesis failed for Episode {episode_number}: {e}",
                exc_info=True,
            )
            # Log details about TTS provider configuration for debugging
            tts_config = self._get_tts_config()
            logger.error(
                f"TTS Provider: {tts_config.provider_name}, "
                f"Voice: {tts_config.voice}, Language: {tts_config.language}"
            )
            base_url = getattr(tts_config, "base_url", None)
            if base_url:
                logger.error(f"TTS API URL: {base_url}")
            raise

    def _split_text_into_chunks(
        self, text: str, max_words: int = _MAX_WORDS_PER_LLM_CHUNK
    ) -> List[str]:
        """Split *text* into chunks of at most *max_words* words.

        Splits preferentially at blank-line (paragraph) boundaries so each
        chunk is a coherent block of prose.  If a single paragraph exceeds
        *max_words*, it is split by word count alone.

        Returns a list with a single element when the text already fits
        within the limit.
        """
        words = text.split()
        if len(words) <= max_words:
            return [text]

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[str] = []
        current_words: List[str] = []
        current_count = 0

        for paragraph in paragraphs:
            para_words = paragraph.split()
            para_count = len(para_words)

            # Flush the current buffer before starting a new paragraph when
            # adding it would exceed the limit.
            if current_count + para_count > max_words and current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_count = 0

            if para_count > max_words:
                # A single oversized paragraph: split by word count.
                for start in range(0, para_count, max_words):
                    chunks.append(" ".join(para_words[start : start + max_words]))
            else:
                current_words.extend(para_words)
                current_count += para_count

        if current_words:
            chunks.append(" ".join(current_words))

        result = chunks if chunks else [text]

        # Lightweight token-size safety net: warn if any chunk is likely to
        # exceed the LLM context window.  Uses the ~4 chars/token heuristic.
        num_ctx = _DEFAULT_LLM_PARAMS["num_ctx"]
        num_predict = _DEFAULT_LLM_PARAMS["num_predict"]
        max_input_tokens = num_ctx - num_predict
        for idx, chunk in enumerate(result):
            estimated_tokens = len(chunk) / 4
            if estimated_tokens > max_input_tokens * 0.9:
                logger.warning(
                    f"LLM chunk {idx + 1}/{len(result)} may exceed context "
                    f"window: ~{estimated_tokens:.0f} estimated tokens vs "
                    f"{max_input_tokens} available input tokens. Consider "
                    "reducing --max-chapters or the source text size."
                )

        return result

    def _break_script_into_segments(self, script: str) -> List[str]:
        """Break script into ≤200-char segments for better TTS chunking."""
        sentences = break_text_into_sentences(script)
        segments: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current + " " + sentence) > 200:
                segments.append(current.strip())
                current = sentence
            else:
                current = (current + " " + sentence) if current else sentence
        if current.strip():
            segments.append(current.strip())
        return [s for s in segments if s.strip()]

    def _save_chapter_titles(self) -> None:
        """Persist chapter titles for synthesize-only mode."""
        path = self.scripts_path / "chapter_titles.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.chapter_titles, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved chapter titles to {path}")
        except IOError as e:
            logger.warning(f"Failed to save chapter titles: {e}")

    def _load_chapter_titles(self) -> list[str]:
        """Load chapter titles from JSON saved during a previous process-only run."""
        path = self.scripts_path / "chapter_titles.json"
        if not path.exists():
            logger.warning("No chapter_titles.json found for synthesize-only mode")
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load chapter titles: {e}")
            return []

    def _validate_chapters(
        self, script_word_counts: list[tuple[str, int]]
    ) -> list[tuple[str, int, str]]:
        """Print a word-count table and flag suspiciously short chapters.

        Returns the flagged entries so callers can decide how to proceed.
        """
        if not script_word_counts:
            return []

        # Print an aligned table header
        sep = "-" * 72
        logger.info("Chapter word-count validation:\n" + sep)
        logger.info(f"{'#':>4}  {'Title':<50} {'Words':>6}  Status")
        logger.info(sep)

        flagged: list[tuple[str, int, str]] = []
        for idx, (title, wc) in enumerate(script_word_counts, 1):
            if wc < 200:
                status = "⚠ SHORT"
                flagged.append((title, wc, f"Chapter {idx}: {wc} words"))
            else:
                status = "OK"
            # Truncate long titles for the table
            display_title = (title[:47] + "...") if len(title) > 50 else title
            logger.info(f"{idx:>4}  {display_title:<50} {wc:>6}  {status}")

        logger.info(sep)

        if flagged:
            summary = "; ".join(entry[2] for entry in flagged)
            logger.warning(
                f"Found {len(flagged)} short chapter(s): {summary}\n"
                "Consider reviewing the chapter titles in the table above. "
                "If they appear to be fragments of a larger chapter, the TOC "
                "grouping may need adjustment."
            )
        else:
            logger.info("All chapters have sufficient content. ✓")

        return flagged

    def _synthesize_from_existing_scripts(self) -> list[Path]:
        """Load saved scripts from the filesystem and synthesize TTS audio.

        Called when *mode* is ``"synthesize"``.  Scans
        ``scripts_path/episode_XXX_script.txt`` files, synthesises each,
        and produces the final M4B.
        """
        self._verify_tts_provider_available()

        script_files = sorted(self.scripts_path.glob("episode_*_script.txt"))
        if not script_files:
            logger.error(
                "No existing scripts found for synthesize-only mode. "
                "Run with --process-only (or no flag) first to generate scripts."
            )
            return []

        episode_paths: list[Path] = []
        self.chapter_titles = self._load_chapter_titles()

        for script_file in script_files:
            # Extract episode number from filename
            try:
                episode_num = int(script_file.stem.split("_")[1])
            except (IndexError, ValueError):
                logger.warning(
                    f"Could not parse episode number from {script_file.name}"
                )
                continue

            try:
                with open(script_file, "r", encoding="utf-8") as f:
                    audiobook_script = f.read()
            except IOError as e:
                logger.warning(f"Could not read {script_file}: {e}")
                continue

            # Progress and title display
            chapter_title = (
                self.chapter_titles[episode_num - 1]
                if episode_num <= len(self.chapter_titles)
                else f"Episode {episode_num}"
            )
            text_snippet = " ".join(audiobook_script.split()[:100])
            self.progress.print_chapter_start(episode_num, chapter_title, text_snippet)

            self.progress.set_phase("Synthesizing")
            episode_path = self.synthesize_episode(audiobook_script, episode_num)

            if episode_path.exists():
                episode_paths.append(episode_path)
                logger.info(
                    f"Successfully created Episode {episode_num}: {episode_path}"
                )
            else:
                logger.warning(f"Failed to create Episode {episode_num}")

        if episode_paths:
            logger.info(
                f"Synthesize-only complete. Created {len(episode_paths)} episodes."
            )
            self.create_m4b()

        return episode_paths

    def create_audiobook_series(self) -> List[Path]:
        """Create audiobook series from all chapters.

        Behaviour depends on ``self.mode``:

        * ``"full"`` (default): extract → generate scripts → validate → synthesise → M4B
        * ``"process"``: extract → generate scripts → validate → save → stop
        * ``"synthesize"``: load saved scripts → synthesise → M4B
        """
        # Synthesize-only short-circuit: load existing scripts and run TTS.
        if self.mode == "synthesize":
            return self._synthesize_from_existing_scripts()

        logger.info("Starting audiobook series creation...")

        # Verify TTS provider before processing, but do not hard-fail unless
        # strict preflight is explicitly enabled.  Skip in process-only mode
        # since no synthesis will happen.
        if self.mode != "process":
            self._verify_tts_provider_available()

        if isinstance(self.reader, EpubReader):
            chapters = self.reader.get_chapters()
        else:
            chapters = [self.reader.cleaned_text]

        num_chapters = len(chapters)
        if self.max_chapters:
            num_chapters = min(num_chapters, self.max_chapters)
            chapters = chapters[:num_chapters]

        logger.info(f"Creating audiobook series with {num_chapters} episodes...")

        # Prepare chapter titles for display
        chapter_titles = []
        for i, chapter_content in enumerate(chapters, 1):
            if isinstance(self.reader, EpubReader):
                title = str(self.reader.get_chapter_title(chapter_content))
            else:
                title = f"Chapter {i}"
            chapter_titles.append(title)

        # Display table of contents
        self.progress.stop()
        self.progress.print_table_of_contents(chapter_titles)
        self.progress.start()

        if self.confirm:
            self.progress.stop()
            response = input(f"Create {num_chapters} audiobook episodes? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                logger.info("Audiobook creation cancelled by user.")
                return []
            self.progress.start()

        # ------------------------------------------------------------------
        # Phase 1 - generate scripts for every chapter
        # ------------------------------------------------------------------
        script_word_counts: list[tuple[str, int]] = []
        chapter_scripts: list[tuple[int, str]] = []

        for i, chapter_content in enumerate(
            track(chapters, description="Creating Audiobook Scripts")
        ):
            episode_number = i + 1
            chapter_title = chapter_titles[i]

            cleaned_content = _clean_text_for_audiobook(chapter_content)
            text_snippet = " ".join(cleaned_content.split()[:100])

            try:
                self.progress.print_chapter_start(
                    episode_number, chapter_title, text_snippet
                )

                self.progress.set_phase("Generating")
                audiobook_script = self.generate_audiobook_script(
                    chapter_content,
                    episode_number,
                )
                chapter_scripts.append((episode_number, audiobook_script))

                # Resumed episodes return "" — exclude them from validation
                # so they don't trigger spurious SHORT warnings.
                if audiobook_script:
                    word_count = len(audiobook_script.split())
                    script_word_counts.append((chapter_title, word_count))

            except Exception as e:
                logger.error(
                    f"Error generating script for Episode {episode_number}: {e}",
                    exc_info=True,
                )
                continue

        # ------------------------------------------------------------------
        # Phase 2 - validate chapter lengths
        # ------------------------------------------------------------------
        self.progress.set_phase("Validating")
        self._validate_chapters(script_word_counts)
        self._save_chapter_titles()

        # In process-only mode we stop before TTS synthesis.
        if self.mode == "process":
            logger.info(
                "Process-only mode: scripts generated and saved. "
                "To synthesise audio, re-run with --synthesize-only."
            )
            return []

        # ------------------------------------------------------------------
        # Phase 3 - synthesise TTS audio
        # ------------------------------------------------------------------
        episode_paths: list[Path] = []

        for episode_number, audiobook_script in chapter_scripts:
            chapter_title = chapter_titles[episode_number - 1]
            text_snippet = " ".join(audiobook_script.split()[:100])

            try:
                self.progress.print_chapter_start(
                    episode_number, chapter_title, text_snippet
                )

                self.progress.set_phase("Synthesizing")
                episode_path = self.synthesize_episode(audiobook_script, episode_number)

                if episode_path.exists():
                    episode_paths.append(episode_path)
                    logger.info(
                        f"Successfully created Episode {episode_number}: {episode_path}"
                    )
                else:
                    logger.warning(f"Failed to create Episode {episode_number}")

            except TTSSynthesisError:
                raise  # TTS failures must not be silently skipped
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

    def _split_episodes_by_duration(
        self, episode_mp3_files: List[Path], max_hours: float = 6.0
    ) -> List[List[Path]]:
        """Split episode MP3 files into chunks with maximum duration in hours."""
        return AudioProcessor.split_audio_by_duration(episode_mp3_files, max_hours)

    def _create_temp_m4b_for_chunk(
        self, chunk_files: List[Path], chunk_index: int
    ) -> Path:
        """Create a temporary M4B file for a specific chunk of episodes."""
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
            f"Combining {len(chunk_files)} episode MP3s for chunk {chunk_index + 1}..."
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
                # Extract episode number from filename: episode_<number>.mp3
                episode_num = int(mp3_file.stem.split("_")[1])
                duration = AudioProcessor.get_duration(str(mp3_file))
                if duration > 0:
                    # Use chapter title if available, otherwise "Episode X"
                    chapter_title = (
                        self.chapter_titles[episode_num - 1]
                        if (episode_num - 1) < len(self.chapter_titles)
                        else None
                    )
                    title = chapter_title or f"Episode {episode_num}"
                    current_start_ms = append_chapter_metadata(
                        chunk_metadata_path,
                        title,
                        current_start_ms,
                        duration,
                    )
            except Exception as e:
                logger.warning(f"Could not process metadata for {mp3_file}: {e}")
        return chunk_metadata_path

    def create_m4b(self) -> None:
        """Combines audiobook episodes into M4B audiobook file."""
        self.progress.set_phase("Combining")
        logger.info("Starting M4B creation process for audiobook...")

        episode_mp3_files = sorted(self.episodes_path.glob("episode_*.mp3"))
        if not episode_mp3_files:
            logger.error("No episode MP3 files found to create M4B.")
            return

        self.temp_m4b_path = self.audiobook_path / f"{self.file_name}.tmp.m4b"
        self.final_m4b_path = self.audiobook_path / f"{self.file_name}.m4b"

        total_duration_hours = self._calculate_total_duration(episode_mp3_files) / 3600
        logger.info(f"Total audiobook duration: {total_duration_hours:.2f} hours")

        # If duration is less than 6 hours, create a single M4B
        # Using 6 hours as a safe limit to avoid WAV file size issues
        if total_duration_hours <= 6.0:
            logger.info("Creating single M4B file (duration <= 6 hours)")
            self._initialize_metadata_file()
            self._create_single_m4b(episode_mp3_files)
        else:
            logger.info(
                f"Duration ({total_duration_hours:.2f}h) exceeds 6 hours, "
                f"splitting into multiple M4B files to avoid WAV file size limits"
            )
            chunks = self._split_episodes_by_duration(episode_mp3_files, max_hours=6.0)
            logger.info(f"Split into {len(chunks)} chunks")

            for chunk_index, chunk_files in enumerate(chunks):
                chunk_duration = self._calculate_total_duration(chunk_files) / 3600
                logger.info(
                    f"Processing chunk {chunk_index + 1}/{len(chunks)} "
                    f"({len(chunk_files)} episodes, {chunk_duration:.2f}h)"
                )

                chunk_temp_path = self._create_temp_m4b_for_chunk(
                    chunk_files, chunk_index
                )
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
                    f"Creating final M4B for chunk {chunk_index + 1}: "
                    f"{chunk_final_path}"
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
                    logger.error(
                        f"M4B creation failed for chunk {chunk_index + 1}: {e}"
                    )

            logger.info(f"Created {len(chunks)} M4B files for long audiobook")

    def _create_single_m4b(self, episode_mp3_files: List[Path]) -> None:
        """Create a single M4B file from all audiobook episodes."""
        if not self.temp_m4b_path.exists():
            logger.info(f"Combining {len(episode_mp3_files)} episode MP3s...")
            combined_audio = AudioSegment.empty()
            current_start_time_ms = 0

            for i, mp3_file in enumerate(
                track(episode_mp3_files, description="Combining Episodes")
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
        try:
            self.progress.start()
            self.progress.set_phase("Reading")
            logger.info(f"Starting audiobook creation for: {self.path.name}")
            episode_paths = self.create_audiobook_series()

            if episode_paths:
                logger.info(
                    f"Audiobook series complete with {len(episode_paths)} episodes"
                )
                return self.audiobook_path

            if self.mode == "process":
                logger.info(
                    "Process-only mode completed. Scripts are saved — "
                    "no audio was synthesized."
                )
            elif self.mode == "synthesize":
                logger.info(
                    "Synthesize-only mode completed — no episodes were produced."
                )
            else:
                logger.error("No audiobook episodes were created successfully")
            return self.audiobook_path
        finally:
            self.progress.stop()


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
        """Create a single audiobook episode from the PDF content.

        Behaviour depends on ``self.mode``:

        * ``"full"`` (default): extract → generate script → synthesise → M4B
        * ``"process"``: extract → generate script → save → validate → stop
        * ``"synthesize"``: load saved script → synthesise → M4B
        """
        logger.info("Creating audiobook episode from PDF...")

        # Synthesize-only short-circuit: load existing script and run TTS.
        if self.mode == "synthesize":
            return self._synthesize_from_existing_scripts()

        # Verify TTS provider before processing, but do not hard-fail unless
        # strict preflight is explicitly enabled.  Skip in process-only mode
        # since no synthesis will happen.
        if self.mode != "process":
            self._verify_tts_provider_available()

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
            self.progress.stop()  # Stop spinner before showing confirmation
            response = input("Create audiobook episode from PDF? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                logger.info("Audiobook creation cancelled by user.")
                return []
            self.progress.start()  # Restart spinner after user confirms

        try:
            logger.info(f"Using language: {self.language}")
            if len(pdf_text.split()) < 200:
                logger.warning(
                    "PDF has very little text. "
                    "The generated audiobook may be very short."
                )

            audiobook_script = self.generate_audiobook_script(pdf_text, 1)

            if self.mode == "process":
                logger.info(
                    "Process-only mode completed. Script is saved — "
                    "no audio was synthesized."
                )
                return []

            episode_path = self.synthesize_episode(audiobook_script, 1)

            if episode_path.exists():
                logger.info(f"Successfully created audiobook episode: {episode_path}")
                return [episode_path]
            else:
                logger.warning("Failed to create audiobook episode")
                return []

        except Exception as e:
            logger.error(f"Error creating audiobook episode: {e}", exc_info=True)
            return []


class DirectoryAudiobookCreator:
    """Creates a single audiobook from multiple files in a directory."""

    def __init_progress(self) -> None:
        """Initialize progress indicator if not already done."""
        if not hasattr(self, "_progress"):
            self._progress = ProgressIndicator()

    @property
    def progress(self) -> ProgressIndicator:
        """Lazy-initialized progress indicator."""
        self.__init_progress()
        return self._progress

    @progress.setter
    def progress(self, value: ProgressIndicator) -> None:
        """Set progress indicator."""
        self._progress = value

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
        mode: str = "full",
    ):
        self.directory_path = Path(directory_path)
        if not self.directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        if mode not in ("full", "process", "synthesize"):
            raise ValueError(
                f"Invalid mode '{mode}': expected 'full', 'process', or 'synthesize'"
            )
        self.mode = mode

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
        self.progress = ProgressIndicator()

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
                    mode=self.mode,
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
                    mode=self.mode,
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

            # Resolve task prompt and metadata
            from audify.prompts.manager import PromptManager
            from audify.prompts.tasks import TaskRegistry

            manager = PromptManager()
            prompt = manager.get_prompt(
                task=self.task or "audiobook",
                prompt_file=self.prompt_file,
            )
            task_config = TaskRegistry.get(self.task or "audiobook")
            llm_params = task_config.llm_params if task_config else {}

            audiobook_script = llm_client.generate_script(
                text=cleaned_content,
                prompt=prompt,
                language=self.language,
                **llm_params,
            )

            if self.save_text:
                script_path = (
                    self.scripts_path / f"episode_{episode_number:03d}_script.txt"
                )
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(audiobook_script)

            title_audio_path = self._synthesize_title_audio(
                article_title, episode_number
            )

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

            combined_audio = AudioSegment.empty()
            if title_audio_path and title_audio_path.exists():
                combined_audio += AudioSegment.from_mp3(title_audio_path)
                combined_audio += AudioSegment.silent(duration=1000)

            combined_audio += AudioSegment.from_mp3(content_mp3_path)

            final_episode_path = (
                self.episodes_path / f"episode_{episode_number:03d}.mp3"
            )
            combined_audio.export(final_episode_path, format="mp3", bitrate="128k")

            logger.info(f"Created episode from text file: {final_episode_path}")
            return final_episode_path

        except Exception as e:
            logger.error(
                f"Error processing text file {file_path.name}: {e}", exc_info=True
            )
            return None

    def _clean_text_for_audiobook(self, text: str) -> str:
        return _clean_text_for_audiobook(text)

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

    def _split_episodes_by_duration(
        self, episode_mp3_files: List[Path], max_hours: float = 6.0
    ) -> List[List[Path]]:
        """Split episode MP3 files into chunks with maximum duration in hours."""
        return AudioProcessor.split_audio_by_duration(episode_mp3_files, max_hours)

    def _create_temp_m4b_for_chunk(
        self, chunk_files: List[Path], chunk_index: int
    ) -> Path:
        """Create a temporary M4B file for a specific chunk of episodes."""
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
            f"Combining {len(chunk_files)} episode MP3s for chunk {chunk_index + 1}..."
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
                # Extract episode number from filename: episode_<number>.mp3
                episode_num = int(mp3_file.stem.split("_")[1])
                duration = AudioProcessor.get_duration(str(mp3_file))
                if duration > 0:
                    # Use chapter title if available, otherwise "Episode X"
                    chapter_title = (
                        self.chapter_titles[episode_num - 1]
                        if (episode_num - 1) < len(self.chapter_titles)
                        else None
                    )
                    title = chapter_title or f"Episode {episode_num}"
                    current_start_ms = append_chapter_metadata(
                        chunk_metadata_path,
                        title,
                        current_start_ms,
                        duration,
                    )
            except Exception as e:
                logger.warning(f"Could not process metadata for {mp3_file}: {e}")
        return chunk_metadata_path

    def create_m4b(self) -> None:
        """Combines episodes into M4B audiobook file."""
        self.progress.set_phase("Combining")
        logger.info("Starting M4B creation process for directory audiobook...")

        # Get all episode MP3 files
        episode_mp3_files = sorted(self.episodes_path.glob("episode_*.mp3"))

        if not episode_mp3_files:
            logger.error("No episode MP3 files found to create M4B.")
            return

        # Set up paths for M4B creation (used for single M4B case)
        self.temp_m4b_path = self.audiobook_path / f"{self.file_name}.tmp.m4b"
        self.final_m4b_path = self.audiobook_path / f"{self.file_name}.m4b"

        total_duration_hours = self._calculate_total_duration(episode_mp3_files) / 3600
        logger.info(f"Total audiobook duration: {total_duration_hours:.2f} hours")

        # If duration is less than 6 hours, create a single M4B
        # Using 6 hours as a safe limit to avoid WAV file size issues
        if total_duration_hours <= 6.0:
            logger.info("Creating single M4B file (duration <= 6 hours)")
            self._initialize_metadata_file()
            self._create_single_m4b(episode_mp3_files)
        else:
            logger.info(
                f"Duration ({total_duration_hours:.2f}h) exceeds 6 hours, "
                f"splitting into multiple M4B files to avoid WAV file size limits"
            )
            chunks = self._split_episodes_by_duration(episode_mp3_files, max_hours=6.0)
            logger.info(f"Split into {len(chunks)} chunks")

            for chunk_index, chunk_files in enumerate(chunks):
                chunk_duration = self._calculate_total_duration(chunk_files) / 3600
                logger.info(
                    f"Processing chunk {chunk_index + 1}/{len(chunks)} "
                    f"({len(chunk_files)} episodes, {chunk_duration:.2f}h)"
                )

                chunk_temp_path = self._create_temp_m4b_for_chunk(
                    chunk_files, chunk_index
                )
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
                    f"Creating final M4B for chunk {chunk_index + 1}: "
                    f"{chunk_final_path}"
                )
                try:
                    assemble_m4b(
                        chunk_temp_path,
                        chunk_metadata_path,
                        chunk_final_path,
                        None,  # No cover image for directory audiobooks
                    )
                    chunk_metadata_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.error(
                        f"M4B creation failed for chunk {chunk_index + 1}: {e}"
                    )

            logger.info(f"Created {len(chunks)} M4B files for long audiobook")

    def _create_single_m4b(self, episode_mp3_files: List[Path]) -> None:
        """Create a single M4B file from all episodes."""
        if not self.temp_m4b_path.exists():
            logger.info(f"Combining {len(episode_mp3_files)} episode MP3s...")
            combined_audio = AudioSegment.empty()
            current_start_time_ms = 0

            for i, mp3_file in enumerate(
                track(episode_mp3_files, description="Combining Episodes")
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
        try:
            self.progress.start()
            self.progress.set_phase("Reading")
            logger.info(
                f"Starting directory audiobook creation for: {self.directory_path.name}"
            )

            # Get all supported files
            files = self._get_supported_files()

            if not files:
                logger.error("No supported files found in directory")
                return self.audiobook_path

            if self.confirm:
                self.progress.stop()  # Stop spinner before showing confirmation
                prompt = (
                    f"Process {len(files)} files from directory "
                    f"'{self.directory_path.name}'? (y/N): "
                )
                response = input(prompt)
                if response.lower() not in ["y", "yes"]:
                    logger.info("Directory audiobook creation cancelled by user.")
                    return self.audiobook_path
                self.progress.start()  # Restart spinner after user confirms

            # Process each file
            for i, file_path in enumerate(files, start=1):
                self.progress.set_phase("Processing")
                episode_path = self._process_single_file(file_path, i)
                if episode_path and episode_path.exists():
                    self.episode_paths.append(episode_path)
                    logger.info(
                        f"Successfully processed file {i}/{len(files)}: "
                        f"{file_path.name}"
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
        finally:
            self.progress.stop()
