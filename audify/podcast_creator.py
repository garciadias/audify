import logging
from pathlib import Path
from typing import List, Optional

import requests
import tqdm

from audify.ebook_read import EpubReader
from audify.pdf_read import PdfReader
from audify.text_to_speech import BaseSynthesizer
from audify.utils import clean_text, get_file_name_title

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODULE_PATH = Path(__file__).resolve().parents[1]
OUTPUT_BASE_DIR = MODULE_PATH / "../" / "data" / "output"

PODCAST_INSTRUCTIONS = """Create an extensive explanation of the content \
shared bellow. All text you generate will be directly read by an AI tool; \
therefore, only text that should be converted to audio should appear in your \
response. Follow a lecture style of speech.

Instructions:
1. Read this content carefully and identify the main concepts and background \
knowledge necessary to understand the developments presented.

2. Create an engaging introduction that:
   - Introduces the topic and its relevance
   - Provides substantial explanations of key concepts needed to understand \
   the material
   - Make it as long as the main content

3. Present the main content with detailed explanations:
   - Highlight what is new or significant in the content
   - Provide in-depth analysis and context
   - Break down complex ideas into digestible segments
   - Use conversational language suitable for audio consumption
   - This should be the most substantial section

4. Conclude by:
   - Summarizing the key takeaways
   - Highlighting the results or implications
   - Suggesting areas for further exploration

Remember: Only include text that should be spoken aloud.
Avoid stage directions, notes, or non-spoken content.
Do not include lists of references, citations, or URLs.
Do not over-summarize the content; provide detailed explanations.

Content:
--------
"""


class LLMClient:
    """Client for interacting with local LLM API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2"
    ):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minute timeout

    def generate_podcast_script(self, chapter_text: str) -> str:
        """Generate podcast script from chapter text using local LLM."""
        prompt = PODCAST_INSTRUCTIONS + "\n\n" + chapter_text

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 4 * 4096,
                # "temperature": 0.9,
                # "top_p": 0.8,
                # "repeat_penalty": 1.1,
                "seed": 428798,
                # "top_k": 50,
                # "min_p": 0.05,
                "num_predict": -1,
            }
        }

        try:
            logger.info(f"Sending request to LLM at {self.base_url}/api/generate")
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()
            if "response" in result:
                # if it is a reasoning model, eliminate the reasoning steps
                if "think" in result["response"].lower():
                    return clean_text(result["response"].split("</think>")[-1].strip())
                return clean_text(result["response"].strip())
            else:
                logger.error(f"Unexpected response format: {result}")
                return "Error: Unable to generate podcast script for this content."

        except requests.exceptions.ConnectionError:
            logger.error(
                f"Could not connect to LLM at {self.base_url}. "
                "Is Ollama running?"
            )
            return (
                "Error: Could not connect to local LLM server. "
                f"Please ensure Ollama is running at {self.base_url}."
            )
        except requests.exceptions.Timeout:
            logger.error("Request to LLM timed out")
            return "Error: Request to LLM timed out. Content might be too long."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with LLM: {e}")
            return f"Error: Failed to generate podcast script due to: {str(e)}"


class PodcastCreator(BaseSynthesizer):
    """Creates podcasts from ebook content using LLM and TTS."""

    def __init__(
        self,
        path: str | Path,
        language: Optional[str] = None,
        speaker: str = "data/Jennifer_16khz.wav",
        model_name: str | None = None,
        translate: Optional[str] = None,
        save_text: bool = True,  # Default to True for podcast scripts
        engine: str = "kokoro",
        llm_base_url: str = "http://localhost:11434",
        llm_model: str = "llama3.2",
        max_chapters: Optional[int] = None,
        confirm: bool = True,
    ):
        # Initialize file reader based on extension
        file_path = Path(path)
        if file_path.suffix.lower() == '.epub':
            self.reader = EpubReader(path)
            detected_language = self.reader.get_language()
            self.title = self.reader.title
        elif file_path.suffix.lower() == '.pdf':
            self.reader = PdfReader(path)
            detected_language = "en"  # PDF reader might not detect language
            self.title = file_path.stem
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        resolved_language = language or detected_language
        if not resolved_language:
            raise ValueError(
                "Language must be provided or detectable from the file metadata."
            )

        # Setup output paths
        self.output_base_dir = Path(OUTPUT_BASE_DIR).resolve()
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Create podcast-specific title
        self.podcast_title = f"Podcast - {self.title}"
        self.file_name = get_file_name_title(self.podcast_title)
        self._setup_paths(self.file_name)

        # Initialize LLM client
        self.llm_client = LLMClient(llm_base_url, llm_model)
        self.max_chapters = max_chapters
        self.confirm = confirm

        # Initialize parent class
        super().__init__(
            path, resolved_language, speaker, model_name, translate, save_text,
            engine
        )

        # Setup cover image if available
        if hasattr(self.reader, 'get_cover_image'):
            self.cover_image_path: Optional[Path] = self.reader.get_cover_image(
                self.podcast_path
            )
        else:
            self.cover_image_path = None

    def _setup_paths(self, file_name_base: str) -> None:
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

    def generate_podcast_script(self, chapter_content: str, chapter_number: int) -> str:
        """Generate podcast script for a single chapter."""
        logger.info(f"Generating podcast script for Chapter {chapter_number}...")

        # Save script path
        script_path = self.scripts_path / f"episode_{chapter_number:03d}_script.txt"

        # Check if script already exists
        if script_path.exists() and not self.confirm:
            logger.info(
                f"Script for Episode {chapter_number} already exists, loading..."
            )
            with open(script_path, 'r', encoding='utf-8') as f:
                return f.read()

        # Extract text from chapter
        if hasattr(self.reader, 'extract_text'):
            chapter_text = self.reader.extract_text(chapter_content)
        else:
            chapter_text = chapter_content

        if not chapter_text.strip():
            logger.warning(f"No text found in Chapter {chapter_number}")
            return "This chapter contains no readable text content."

        # Generate podcast script using LLM
        podcast_script = self.llm_client.generate_podcast_script(chapter_text)

        # Save the script if requested
        if self.save_text:
            try:
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Podcast Episode {chapter_number}\n")
                    f.write(f"# Generated from: {self.title}\n\n")
                    f.write(podcast_script)
                logger.info(f"Saved podcast script to: {script_path}")
            except IOError as e:
                logger.error(
                    f"Failed to save script for Episode {chapter_number}: {e}"
                )

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
            logger.warning(
                f"Episode {episode_number} contains no text to synthesize."
            )
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
                f"Translating {len(sentences)} segments for Episode "
                f"{episode_number}..."
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
        if hasattr(self.reader, 'get_chapters'):
            chapters = self.reader.get_chapters()
        else:
            # For PDF, treat the whole document as one episode
            chapters = [self.reader.get_cleaned_text()]

        num_chapters = len(chapters)
        if self.max_chapters:
            num_chapters = min(num_chapters, self.max_chapters)
            chapters = chapters[:num_chapters]

        logger.info(f"Creating podcast series with {num_chapters} episodes...")

        if self.confirm:
            response = input(f"Create {num_chapters} podcast episodes? (y/N): ")
            if response.lower() not in ['y', 'yes']:
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
                    chapter_content, episode_number
                )

                # Synthesize episode
                episode_path = self.synthesize_episode(
                    podcast_script, episode_number
                )

                if episode_path.exists():
                    episode_paths.append(episode_path)
                    logger.info(
                        f"Successfully created Episode {episode_number}: "
                        f"{episode_path}"
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
            f"Podcast series creation complete. "
            f"Created {len(episode_paths)} episodes."
        )
        return episode_paths

    def synthesize(self) -> Path:
        """Main synthesis method - creates the podcast series."""
        logger.info(f"Starting podcast creation for: {self.path.name}")
        episode_paths = self.create_podcast_series()

        if episode_paths:
            logger.info(
                f"Podcast series complete with {len(episode_paths)} episodes"
            )
            return self.podcast_path
        else:
            logger.error("No podcast episodes were created successfully")
            return self.podcast_path


class PodcastEpubCreator(PodcastCreator):
    """Specialized podcast creator for EPUB files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.reader, 'get_chapters'):
            raise ValueError("PodcastEpubCreator requires an EPUB reader")


class PodcastPdfCreator(PodcastCreator):
    """Specialized podcast creator for PDF files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.reader, 'get_cleaned_text'):
            raise ValueError("PodcastPdfCreator requires a PDF reader")

    def create_podcast_series(self) -> List[Path]:
        """Create a single podcast episode from the PDF content."""
        logger.info("Creating podcast episode from PDF...")

        # Get the full PDF content
        pdf_text = self.reader.get_cleaned_text()

        if not pdf_text.strip():
            logger.error("No text found in PDF")
            return []

        if self.confirm:
            response = input("Create podcast episode from PDF? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("Podcast creation cancelled by user.")
                return []

        try:
            # Generate podcast script
            podcast_script = self.generate_podcast_script(pdf_text, 1)

            # Synthesize episode
            episode_path = self.synthesize_episode(podcast_script, 1)

            if episode_path.exists():
                logger.info(f"Successfully created podcast episode: {episode_path}")
                return [episode_path]
            else:
                logger.warning("Failed to create podcast episode")
                return []

        except Exception as e:
            logger.error(f"Error creating podcast episode: {e}", exc_info=True)
            return []
