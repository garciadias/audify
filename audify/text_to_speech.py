# %%
import subprocess
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from ebooklib.epub import EpubBook
from pydub import AudioSegment
from TTS.api import TTS

from audify.ebook_read import BookReader

MODULE_PATH = Path(__file__).parents[1]

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
LOADED_MODEL = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
LOADED_MODEL.to(device)
# %%


class TextToSpeech:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def sentence_to_speech(
        self,
        sentence: str,
        file_path: str = "tmp/speech.wav",
        language: str = "es",
        speaker: str | Path = "data/Jennifer_16khz.wav",
    ) -> None:
        if Path(file_path).parent.is_dir() is False:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.model.tts_to_file(
                text=sentence,
                file_path=file_path,
                language=language,
                speaker_wav=speaker,
            )
        except Exception as e:
            error_message = "Error: " + str(e)
            self.model.tts_to_file(
                text=error_message,
                file_path=file_path,
                language=language,
                speaker_wav=speaker,
            )

    def chapter_to_speech(
        self,
        sentences: list[str],
        chapter_number: int,
        audiobook_path: str | Path,
        language: str,
    ) -> None:
        self.sentence_to_speech(
            sentence=sentences[0],
            file_path=f"{audiobook_path}/chapter_{chapter_number}.wav",
            language=language,
        )
        combined_audio = AudioSegment.from_wav(
            f"{audiobook_path}/chapter_{chapter_number}.wav"
        )
        for sentence in sentences[1:]:
            self.sentence_to_speech(sentence=sentence, language=language)
            audio = AudioSegment.from_wav("tmp/speech.wav")
            combined_audio += audio
            combined_audio.export(
                f"{audiobook_path}/chapter_{chapter_number}.wav", format="wav"
            )

    def create_m4b(self, chapter_files, filename, cover_image_path: str | None = None):
        tmp_filename = filename.replace(".epub", ".tmp.mp4")
        if not Path(tmp_filename).exists():
            combined_audio = AudioSegment.empty()
            for wav_file in chapter_files:
                audio = AudioSegment.from_wav(wav_file)
                combined_audio += audio
            print("Converting to M4b...")
            combined_audio.export(
                tmp_filename, format="mp4", codec="aac", bitrate="64k"
            )
        final_filename = filename.replace(".epub", ".m4b")
        print("Creating M4B file...")

        if cover_image_path is not None:
            with open(cover_image_path, "rb") as f:
                cover_image = f.read()
        else:
            cover_image = None
        if cover_image:
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
                f"{cover_image_path.parent}/chapters.txt",
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

    def get_wav_duration(self, file_path):
        # Open the .wav file
        with wave.open(file_path, "rb") as wav_file:
            # Get the number of frames
            frames = wav_file.getnframes()
            # Get the frame rate (samples per second)
            frame_rate = wav_file.getframerate()
            # Calculate the duration in seconds
            duration = frames / float(frame_rate)
            return duration

    def log_on_chapter_file(
        self, chapter_file_path: Path | str, title: str, start: int, duration: float
    ) -> int:
        end = start + int(duration * 1000)
        if isinstance(chapter_file_path, str):
            chapter_file_path = Path(chapter_file_path)
        with open(chapter_file_path.parent / "chapters.txt", "a") as f:
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start}\n")
            f.write(f"END={end}\n")
            f.write(f"TITLE={title}\n")
            f.write("\n")
        return end

    def process_chapter(self, i, chapter, chapter_start, audiobook_path, language):
        if len(chapter) < 1000 or Path(f"{audiobook_path}/chapter_{i}.txt").exists():
            if Path(f"{audiobook_path}/chapter_{i}.txt").exists():
                duration = self.get_wav_duration(f"{audiobook_path}/chapter_{i}.wav")
                chapter_start += int(duration * 1000)
            return chapter_start
        print(f"Synthesizing chapter: {i}")
        self.synthesize_chapter(chapter, i, audiobook_path, language)
        title = f"Chapter {i}: {ebook_read.get_chapter_title(chapter)}"
        duration = self.get_wav_duration(f"{audiobook_path}/chapter_{i}.wav")
        chapter_start = self.log_on_chapter_file(
            f"{audiobook_path}/chapter_{i}.txt", title, chapter_start, duration
        )
        return chapter_start


def process_chapters(book: BookReader, audiobook_path: str | Path) -> None:
    chapter_start = 0
    chapter_id = 1
    chapters = book.read_chapters()
    language = book.get_language()
    synthesizer = TextToSpeech(LOADED_MODEL, device)
    for chapter in chapters:
        if len(chapter) < 1000:
            continue
        chapter_start = synthesizer.process_chapter(
            chapter_id, chapter, chapter_start, audiobook_path, language
        )
        chapter_id += 1
