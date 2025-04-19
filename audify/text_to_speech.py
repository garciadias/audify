import contextlib
import logging
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import soundfile as sf
import torch
import tqdm
from kokoro import KPipeline
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from TTS.api import TTS

from audify.constants import LANG_CODES
from audify.domain.interface import Synthesizer
from audify.ebook_read import EpubReader
from audify.pdf_read import PdfReader
from audify.translate import translate_sentence
from audify.utils import (
    break_text_into_sentences,
    get_audio_duration,
    get_file_name_title,
    sentence_to_speech,
)

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


class BaseSynthesizer(Synthesizer):
    def __init__(
        self,
        path: str | Path,
        language: str | None,
        speaker: str,
        model_name: str,
        translate: str | None,
        save_text: bool,
        engine: str,
    ):
        self.path = path if isinstance(path, Path) else Path(path)
        self.language = language
        self.speaker = speaker
        self.translate = translate
        self.engine = engine
        self.model_name = model_name
        self.save_text = save_text
        self.tmp_dir = Path(f"/tmp/audify/{self.path.stem}/")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading TTS model...")
        self.model = TTS(model_name=model_name)
        self.model.to(device)

    def _synthesize_sentences(self, sentences, output_path):
        combined_audio = AudioSegment.empty()
        if self.engine == "kokoro":
            lang_code = (
                LANG_CODES[self.translate]
                if self.translate
                else LANG_CODES[self.language]
            )
            self.pipeline = KPipeline(lang_code=lang_code)
            generator = self.pipeline(
                sentences,
                voice="af_heart",
                speed=1,
                split_pattern=r"\n+",
            )
            i = 0
            for i, (_, _, audio) in tqdm.tqdm(
                enumerate(generator), desc="Synthesizing audio..."
            ):
                sf.write(f"{self.tmp_dir}/{i}.wav", audio, 24000)
            n_files = i + 1
            for i in range(n_files):
                if not Path(f"{self.tmp_dir}/{i}.wav").exists():
                    continue
                audio = AudioSegment.from_wav(f"{self.tmp_dir}/{i}.wav")
                combined_audio += audio
            combined_audio.export(output_path, format="wav")
        elif self.engine == "tts_models":
            for sentence in tqdm.tqdm(sentences, desc="Synthesizing sentences..."):
                with nostdout():
                    sentence_to_speech(
                        sentence=sentence,
                        output_dir=self.tmp_dir,
                        language=self.language
                        if not self.translate
                        else self.translate,
                        speaker=self.speaker,
                        model=self.model,
                        file_name="speech.wav",
                    )
                audio = AudioSegment.from_wav(self.tmp_dir / "speech.wav")
                combined_audio += audio
            combined_audio.export(output_path, format="wav")

    def _convert_to_mp3(self, wav_path):
        mp3_path = str(wav_path).replace(".wav", ".mp3")
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3")
        Path(wav_path).unlink(missing_ok=True)
        return mp3_path


class EpubSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        path: str | Path,
        language: str | None = None,
        speaker: str = "data/Jennifer_16khz.wav",
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        translate: str | None = None,
        save_text: bool = False,
        engine: str = "kokoro",
        confirm=True,
    ):
        reader = EpubReader(path)
        language = language or reader.get_language()
        title = reader.title
        if translate:
            title = translate_sentence(
                sentence=title, src_lang=language, tgt_lang=translate
            )
        file_name = get_file_name_title(title)
        audiobook_path = Path(f"{MODULE_PATH}/data/output/{file_name}")
        audiobook_path.mkdir(parents=True, exist_ok=True)
        super().__init__(
            path, language, speaker, model_name, translate, save_text, engine
        )
        self.reader = reader
        self.title = title
        self.file_name = file_name
        self.audiobook_path = audiobook_path
        self.list_of_contents = self.audiobook_path / "chapters.txt"
        self.cover_image = self.reader.get_cover_image(self.audiobook_path)
        self.confirm = confirm

        with open(self.list_of_contents, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write("major_brand=M4A\n")
            f.write("minor_version=512\n")
            f.write("compatible_brands=M4A isis2\n")
            f.write("encoder=Lavf61.7.100\n")

    def synthesize_chapter(self, chapter, chapter_number):
        chapter_txt = self.reader.extract_text(chapter)
        sentences = break_text_into_sentences(chapter_txt)
        if self.translate:
            sentences = [
                translate_sentence(
                    sentence, src_lang=self.language, tgt_lang=self.translate
                )
                for sentence in tqdm.tqdm(sentences, desc="Translating sentences...")
            ]
        chapter_path = f"{self.audiobook_path}/chapter_{chapter_number}.wav"
        self._synthesize_sentences(sentences, chapter_path)
        return self._convert_to_mp3(chapter_path)

    def create_m4b(self):
        n_chapters = len(list(self.audiobook_path.glob("chapter*.mp3")))
        chapter_files = [
            f"{self.audiobook_path}/chapter_{i}.mp3" for i in range(1, n_chapters + 1)
        ]
        tmp_file_name = f"{self.audiobook_path}/{self.file_name}.tmp.m4b"
        final_file_name = f"{self.audiobook_path}/{self.file_name}.m4b"
        combined_audio = AudioSegment.empty()
        for mp3 in tqdm.tqdm(chapter_files, desc="Combining chapters..."):
            try:
                audio = AudioSegment.from_mp3(mp3)
                combined_audio += audio
            except CouldntDecodeError:
                logger.error(f"Could not decode file: {mp3}")
                continue
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
        subprocess.run(command)

    def log_on_chapter_file(self, chapter_file_path, title, start, duration):
        end = start + int(duration * 1000)
        with open(self.list_of_contents, "a") as f:
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start}\n")
            f.write(f"END={end}\n")
            f.write(f"title={title}\n")
        return end

    def process_chapter(self, i, chapter, chapter_start):
        is_too_short = len(chapter) < 100
        chapter_path = f"{self.audiobook_path}/chapter_{i}.wav"
        chapter_exists = Path(chapter_path.replace("wav", "mp3")).exists()
        if Path(chapter_path).exists():
            Path(chapter_path).unlink(missing_ok=True)
        chapter_title = self.reader.get_chapter_title(chapter)
        title = f"Chapter {i}: {chapter_title}"
        if is_too_short or chapter_exists:
            if chapter_exists:
                duration = get_audio_duration(chapter_path.replace("wav", "mp3"))
                chapter_start += int(duration * 1000)
                chapter_start = self.log_on_chapter_file(
                    self.list_of_contents, title, chapter_start, duration
                )
            return chapter_start
        else:
            self.synthesize_chapter(chapter, i)
            duration = get_audio_duration(chapter_path.replace(".wav", ".mp3"))
            chapter_start = self.log_on_chapter_file(
                self.list_of_contents, title, chapter_start, duration
            )
        return chapter_start

    def process_chapters(self):
        chapter_start = 0
        chapter_id = 1
        chapters = self.reader.get_chapters()
        self.check_job_proposition()
        for chapter in tqdm.tqdm(chapters, desc=f"Processing {len(chapters)} chapters"):
            if len(chapter) < 100:
                continue
            else:
                chapter_start = self.process_chapter(chapter_id, chapter, chapter_start)
                chapter_id += 1

    def synthesize(self):
        self.process_chapters()
        self.create_m4b()
        return f"{self.audiobook_path}/{self.file_name}"

    def check_job_proposition(self):
        if self.language is None:
            language_from_file = self.reader.get_language()
            print(
                f"Language detected: {language_from_file}."
                " Do you want to use this language? (y/n)"
            )
            use_language = input("Use detected language? (y/n): [y] ")
            if use_language.lower() in ["n", "no"]:
                self.language = input("Enter the language code: ")

        terminal_width = shutil.get_terminal_size((80, 20)).columns
        print("=" * terminal_width)
        print(f"Processing book: {self.title}")
        print("=" * terminal_width)
        print("Job details:")
        print(f"Original file: {self.path.stem}")
        print(f"Title: {self.title}")
        print(f"Language: {self.language}")
        print(f"Speaker: {self.speaker}")
        print(f"Output: {self.audiobook_path}")
        if self.translate:
            print(f"Translate to: {self.translate}")
        if self.confirm:
            confirmation = input("Do you want to proceed? (y/n): [y] ")
        else:
            confirmation = "y"
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


class PdfSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        pdf_path: str | Path,
        language: str = "en",
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        speaker: str = "data/Jennifer_16khz.wav",
        output_dir: str | Path = "data/output/articles/",
        file_name: str | None = None,
        translate: str | None = None,
        save_text: bool = False,
        engine: str = "kokoro",
    ):
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        output_file = Path(output_dir).resolve() / (file_name or pdf_path.stem)
        output_file = output_file.with_suffix(".wav")
        super().__init__(
            pdf_path, language, speaker, model_name, translate, save_text, engine
        )
        self.pdf_path = pdf_path
        self.output_file = output_file

    def synthesize(self):
        reader = PdfReader(self.pdf_path)
        cleaned_text = reader.get_cleaned_text()
        sentences = break_text_into_sentences(cleaned_text)
        if self.translate:
            sentences = [
                translate_sentence(
                    sentence, src_lang=self.language, tgt_lang=self.translate
                )
                for sentence in sentences
            ]
        self._synthesize_sentences(sentences, self.output_file)
        return self._convert_to_mp3(self.output_file)

    @classmethod
    def from_config(cls, config: dict):
        defaults = {
            "output_dir": "outputs",
            "output_name": "output.wav",
            "language": "en",
            "speech_rate": 1.0,
        }
        params = {**defaults, **config}
        return cls(pdf_path=params["pdf_path"])


class InspectSynthesizer(Synthesizer):
    def __init__(
        self,
        path: str | Path = "./",
        language: str | None = None,
        speaker: str = "data/Jennifer_16khz.wav",
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    ):
        self.reader = Path(path)
        self.model_name = model_name
        self.speaker = speaker
        self.language = language or "en"
        self.model = TTS(model_name=model_name)

    def synthesize(self):
        return (
            "This class is used to inspect model options and is not intended "
            "for synthesis."
        )
