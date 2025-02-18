# %%
import contextlib
import logging
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import tqdm
from pydub import AudioSegment
from TTS.api import TTS

from audify.domain.interface import Synthesizer
from audify.ebook_read import EpubReader
from audify.pdf_read import PdfReader
from audify.translate import translate_sentence
from audify.utils import break_text_into_sentences, get_mp3_duration, sentence_to_speech

logger = logging.getLogger(__name__)
MODULE_PATH = Path(__file__).parents[1]

# mute FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = open("trash", "w")
    yield
    sys.stdout = save_stdout


class EpubSynthesizer(Synthesizer):

    def __init__(
        self,
        path: str | Path,
        language: str | None = None,
        speaker: str = "data/Jennifer_16khz.wav",
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        translate: str | None = None,
    ):
        self.reader = EpubReader(path)
        self.language = language or self.reader.get_language()
        self.speaker = speaker
        self.title = self.reader.title
        self.file_name = self.reader.get_file_name_title()
        self.tmp_dir = Path(f"/tmp/audify/{self.file_name}/")
        self.audiobook_path: str | Path = Path(
            f"{MODULE_PATH}/data/output/{self.file_name}"
        )
        self.audiobook_path.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.list_of_contents = self.audiobook_path / "chapters.txt"
        self.cover_image = self.reader.get_cover_image()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the TTS model
        # Mute terminal outputs from TTS
        print("Loading TTS model...")
        self.model = TTS(
            model_name=model_name,
        )
        self.model.to(device)
        self.translate = translate

        with open(self.list_of_contents, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write("major_brand=M4A\n")
            f.write("minor_version=512\n")
            f.write("compatible_brands=M4A isis2\n")
            f.write("encoder=Lavf61.7.100\n")

    def synthesize_chapter(
        self,
        chapter: str,
        chapter_number: int,
        audiobook_path: str | Path,
    ) -> None:
        chapter_txt = self.reader.extract_text(chapter)
        sentences = break_text_into_sentences(chapter_txt)
        if self.translate:
            translated_sentences = []
            for sentence in sentences:
                sentence = translate_sentence(
                    sentence=sentence, src_lang=self.language, tgt_lang=self.translate
                )
                translated_sentences.append(sentence)
            sentences = translated_sentences
        chapter_path = f"{audiobook_path}/chapter_{chapter_number}.mp3"
        announcement = (
            f"Chapter {chapter_number}: " f"{self.reader.get_chapter_title(chapter)}"
        )
        with nostdout():
            self.model.tts_to_file(
                text=announcement,
                file_path=chapter_path,
                language=self.language,
                speaker_wav=self.speaker,
            )
        if Path().exists():
            combined_audio = AudioSegment.from_mp3(chapter_path)
        for sentence in tqdm.tqdm(
            sentences[1:], desc=f"Synthesizing chapter {chapter_number}..."
        ):
            with nostdout():
                sentence_to_speech(
                    sentence=sentence,
                    tmp_dir=self.tmp_dir,
                    language=self.language if not self.translate else self.translate,
                    speaker=self.speaker,
                    model=self.model,
                )
            audio = AudioSegment.from_mp3(self.tmp_dir / "speech.mp3")
            combined_audio += audio
            combined_audio.export(chapter_path, format="mp3")

    def create_m4b(self):
        chapter_files = list(Path(self.audiobook_path).rglob("*.mp3"))
        tmp_file_name = f"{self.audiobook_path}/{self.file_name}.tmp.m4b"
        final_file_name = f"{self.audiobook_path}/{self.file_name}.m4b"
        combined_audio = AudioSegment.empty()
        for mp3_file in tqdm.tqdm(chapter_files, desc="Combining chapters..."):
            audio = AudioSegment.from_mp3(mp3_file)
            combined_audio += audio
        print("Converting to M4b...")
        if not Path(tmp_file_name).exists():
            combined_audio.export(
                tmp_file_name, format="mp4", codec="aac", bitrate="64k"
            )
        print("Adding M4B file metadata...")

        if self.cover_image:
            with open(self.cover_image, "rb") as f:
                cover_image = f.read()

        if self.cover_image:
            cover_image_file = NamedTemporaryFile("wb")
            cover_image_file.write(cover_image)
            cover_image_args = [
                "-i",
                cover_image_file.name,
                "-map",
                "0:a",
                "-map",
                "2:v",
            ]
        else:
            cover_image_args = []
        command = [
            "ffmpeg",
            "-i",
            f"{tmp_file_name}",
            "-i",
            self.list_of_contents,
            *cover_image_args,
            "-map",
            "0",
            "-map_metadata",
            "1",
            "-c:a",
            "copy",
            "-c:v",
            "copy",
            "-disposition:v",
            "attached_pic",
            "-c",
            "copy",
            "-f",
            "mp4",
            f"{final_file_name}",
        ]
        command = [str(arg) for arg in command]
        print(" ".join(command))
        subprocess.run(
            command,
        )

    def log_on_chapter_file(
        self, chapter_file_path: Path | str, title: str, start: int, duration: float
    ) -> int:
        end = start + int(duration * 1000)
        if isinstance(chapter_file_path, str):
            chapter_file_path = Path(chapter_file_path)
        with open(self.list_of_contents, "a") as f:
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start}\n")
            f.write(f"END={end}\n")
            f.write(f"title={title}\n")
        return end

    def process_chapter(self, i, chapter, chapter_start):
        is_too_short = len(chapter) < 1000
        chapter_path = f"{self.audiobook_path}/chapter_{i}.mp3"
        chapter_exists = Path(chapter_path).exists()
        chapter_title = self.reader.get_chapter_title(chapter)
        title = f"Chapter {i}: {chapter_title}"
        if is_too_short or chapter_exists:
            if chapter_exists:
                duration = get_mp3_duration(chapter_path)
                chapter_start += int(duration * 1000)
                chapter_start = self.log_on_chapter_file(
                    self.list_of_contents, title, chapter_start, duration
                )
            return chapter_start
        else:
            self.synthesize_chapter(chapter, i, self.audiobook_path)
            duration = get_mp3_duration(chapter_path)
            chapter_start = self.log_on_chapter_file(
                self.list_of_contents, title, chapter_start, duration
            )
        return chapter_start

    def process_chapters(self) -> None:
        chapter_start = 0
        chapter_id = 1
        chapters = self.reader.get_chapters()
        self.language = self.reader.get_language()
        self.check_job_proposition()
        for chapter in tqdm.tqdm(chapters, desc=f"Processing {len(chapters)} chapters"):
            if len(chapter) < 1000:
                continue
            else:
                chapter_start = self.process_chapter(chapter_id, chapter, chapter_start)
                chapter_id += 1

    def synthesize(self) -> str:
        self.process_chapters()
        self.create_m4b()
        return f"{self.audiobook_path}/{self.file_name}"

    def check_job_proposition(self) -> None:
        if self.language not in ["es", "en", "pt"]:
            self.language = input("Enter the language code: ")
            print(f"Confirm language: You are using {self.language}")

        print(
            "=========================================================================="
        )
        print(f"Processing book: {self.title}")
        print(
            "=========================================================================="
        )
        print("Confirm details:")
        print(f"Title: {self.title}")
        print(f"Language: {self.language}")
        print(f"Speaker: {self.speaker}")
        print(f"Output: {self.audiobook_path}")
        confirmation = input("Do you want to proceed? (y/n): [y] ")
        if confirmation.lower() not in ["y", "yes", ""]:
            self.title = input("Enter the title: ") or self.title
            self.language = input("Enter the language code: ") or self.language
            self.speaker = input("Enter the speaker: ") or self.speaker
            self.audiobook_path = (
                input("Enter the output path: ") or self.audiobook_path
            )
            abort = input("Abort? (y/n): [n] ")
            if abort.lower() in ["y", "yes"]:
                sys.exit(0)
            self.check_job_proposition()


class PdfSynthesizer(Synthesizer):
    def __init__(
        self,
        pdf_path: str | Path,
        language: str = "en",
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        speaker: str = "data/Jennifer_16khz.mp3",
        output_dir: str | Path = "data/output/articles/",
        file_name: str | None = None,
        translate: str | None = None,
    ):
        self.pdf_path = Path(pdf_path).resolve()
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        self.language = language
        self.model_name = model_name
        self.speaker = speaker
        self.output_dir = Path(output_dir).resolve()
        self.output_file = self.output_dir / (file_name or self.pdf_path.stem)
        self.output_file = self.output_file.with_suffix(".mp3")
        self.tmp_dir = Path("/tmp/audify/") / (file_name or self.pdf_path.stem)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.translate = translate

    def _setup_tts(self) -> TTS:
        """Initialize TTS engine with appropriate language model"""
        try:
            logger.info(f"Loading TTS model: {self.model_name}")
            model = TTS(model_name=self.model_name)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {str(e)}")
            raise

    def synthesize(self) -> Path | str:
        """Synthesize PDF content into audio file using TTS"""
        # Extract and clean text from PDF
        reader = PdfReader(self.pdf_path)
        cleaned_text = reader.get_cleaned_text()

        # Setup TTS engine
        self.model = self._setup_tts()

        # Prepare output path
        logger.info(f"Generating audio file: {self.output_file}")

        # Synthesize text into audio
        self._synthesize_clean_text(cleaned_text)
        return self.output_file

    def _synthesize_clean_text(
        self,
        cleaned_text: str,
    ) -> None:
        sentences = break_text_into_sentences(cleaned_text)
        if self.translate:
            translated_sentences = []
            for sentence in sentences:
                sentence = translate_sentence(
                    sentence=sentence, src_lang=self.language, tgt_lang=self.translate
                )
                translated_sentences.append(sentence)
            sentences = translated_sentences
        announcement = "Generated audio file from PDF: " f"{self.pdf_path.stem}"
        with nostdout():
            self.model.tts_to_file(
                text=announcement,
                file_path=self.output_file,
                language=self.language,
                speaker_wav=self.speaker,
            )
        if Path().exists():
            combined_audio = AudioSegment.from_mp3(self.output_file)
        for sentence in tqdm.tqdm(
            sentences[1:], desc=f"Synthesizing file {self.pdf_path.stem}..."
        ):
            with nostdout():
                sentence_to_speech(
                    sentence=sentence,
                    tmp_dir=self.tmp_dir,
                    language=self.language if not self.translate else self.translate,
                    speaker=self.speaker,
                    model=self.model,
                )
            audio = AudioSegment.from_mp3(self.tmp_dir / "speech.mp3")
            combined_audio += audio
            combined_audio.export(self.output_file, format="mp3")

    @classmethod
    def from_config(cls, config: dict):
        """Create instance from configuration dictionary"""
        defaults = {
            "output_dir": "outputs",
            "output_name": "output.mp3",
            "language": "en",
            "speech_rate": 1.0,
        }
        params = {**defaults, **config}
        instance = cls(pdf_path=params["pdf_path"])
        return instance


class InspectSynthesizer(Synthesizer):
    def __init__(
        self,
        path: str | Path = "./",
        language: str | None = None,
        speaker: str = "data/Jennifer_16khz.mp3",
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    ):
        self.reader = Path(path)
        self.model_name = model_name
        self.speaker = speaker
        self.language = language or "en"
        self.model = TTS(model_name=model_name)

    def synthesize(self) -> str:
        return (
            "This class is used to inspect model options and is not intended "
            "for synthesis."
        )
