# %%
import contextlib
import subprocess
import sys
import warnings
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import tqdm
from pydub import AudioSegment
from TTS.api import TTS

from audify.domain.interface import Synthesizer
from audify.ebook_read import EpubReader

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
    ):
        self.reader = EpubReader(path)
        self.language = language or self.reader.get_language()
        self.speaker = speaker
        self.title = self.reader.title
        self.filename = self.reader.get_file_name_title()
        self.tmp_dir = Path(f"/tmp/audify/{self.filename}/")
        self.audiobook_path = Path(f"{MODULE_PATH}/data/output/{self.filename}")
        self.audiobook_path.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.list_of_contents = self.audiobook_path / "chapters.txt"
        self.cover_image = self.reader.get_cover_image()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the TTS model
        # Mute terminal outputs from TTS
        print("Loading TTS model...")
        with nostdout():
            self.model = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            )
        self.model.to(device)

        with open(self.list_of_contents, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write("major_brand=M4A\n")
            f.write("minor_version=512\n")
            f.write("compatible_brands=M4A isis2\n")
            f.write("encoder=Lavf61.7.100\n")

    def sentence_to_speech(self, sentence: str) -> None:
        if Path(self.tmp_dir).parent.is_dir() is False:
            Path(self.tmp_dir).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.model.tts_to_file(
                text=sentence,
                file_path=self.tmp_dir / "speech.wav",
                language=self.language,
                speaker_wav=self.speaker,
            )
        except Exception as e:
            error_message = "Error: " + str(e)
            self.model.tts_to_file(
                text=error_message,
                file_path=self.tmp_dir / "speech.wav",
                language=self.language,
                speaker_wav=self.speaker,
            )

    def synthesize_chapter(
        self,
        chapter: str,
        chapter_number: int,
        audiobook_path: str | Path,
    ) -> None:
        chapter_txt = self.reader.extract_text(chapter)
        sentences = self.reader.break_text_into_sentences(chapter_txt)
        chapter_path = f"{audiobook_path}/chapter_{chapter_number}.wav"
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
            combined_audio = AudioSegment.from_wav(chapter_path)
        for sentence in tqdm.tqdm(
            sentences[1:], desc=f"Synthesizing chapter {chapter_number}..."
        ):
            with nostdout():
                self.sentence_to_speech(sentence=sentence)
            audio = AudioSegment.from_wav(self.tmp_dir / "speech.wav")
            combined_audio += audio
            combined_audio.export(chapter_path, format="wav")

    def create_m4b(self):
        chapter_files = list(Path(self.audiobook_path).rglob("*.wav"))
        tmp_filename = self.filename.replace(".epub", ".tmp.mp4")
        if not Path(tmp_filename).exists():
            combined_audio = AudioSegment.empty()
            for wav_file in chapter_files:
                audio = AudioSegment.from_wav(wav_file)
                combined_audio += audio
            print("Converting to M4b...")
            combined_audio.export(
                tmp_filename, format="mp4", codec="aac", bitrate="64k"
            )
        final_filename = self.filename.replace(".epub", ".m4b")
        print("Creating M4B file...")

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
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                f"{tmp_filename}",
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
                f"{final_filename}",
            ]
        )
        Path(tmp_filename).unlink()

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
            f.write(f"TITLE={title}\n")
            f.write("\n")
        return end

    def process_chapter(self, i, chapter, chapter_start):
        is_too_short = len(chapter) < 1000
        chapter_path = f"{self.audiobook_path}/chapter_{i}.wav"
        chapter_exists = Path(chapter_path).exists()
        if is_too_short or chapter_exists:
            if chapter_exists:
                duration = get_wav_duration(chapter_path)
                chapter_start += int(duration * 1000)
            return chapter_start
        else:
            chapter_title = self.reader.get_chapter_title(chapter)
            self.synthesize_chapter(chapter, i, self.audiobook_path)
            title = f"Chapter {i}: {chapter_title}"
            duration = get_wav_duration(chapter_path)
            chapter_start = self.log_on_chapter_file(
                self.list_of_contents, title, chapter_start, duration
            )
        return chapter_start

    def process_chapters(self) -> None:
        chapter_start = 0
        chapter_id = 1
        chapters = self.reader.get_chapters()
        self.language = self.reader.get_language()
        if self.language not in ["es", "en", "pt"]:
            self.language = input("Enter the language code: ")
            print(f"Using language: {self.language}")

        print("=====================================")
        print(f"Processing book: {self.title}")
        print("=====================================")
        print("Confirm details:")
        print(f"Title: {self.title}")
        print(f"Language: {self.language}")
        print(f"Speaker: {self.speaker}")
        print(f"Output: {self.audiobook_path}")
        confirmation = input("Do you want to proceed? (y/n): [y]")
        if confirmation.lower() not in ["y", "yes", ""]:
            print("Exiting...")
            sys.exit(0)
        for chapter in tqdm.tqdm(chapters, desc=f"Processing {len(chapters)} chapters"):
            if len(chapter) < 1000:
                continue
            else:
                chapter_start = self.process_chapter(chapter_id, chapter, chapter_start)
                chapter_id += 1

    def synthesize(self) -> str:
        self.process_chapters()
        self.create_m4b()
        return f"{self.audiobook_path}/{self.filename}"


def get_wav_duration(file_path):
    # Open the .wav file
    with wave.open(file_path, "rb") as wav_file:
        # Get the number of frames
        frames = wav_file.getnframes()
        # Get the frame rate (samples per second)
        frame_rate = wav_file.getframerate()
        # Calculate the duration in seconds
        duration = frames / float(frame_rate)
        return duration
