import subprocess
import tempfile
import wave
from pathlib import Path

import torch
from pydub import AudioSegment
from TTS.api import TTS

from audify.ebook_read import BookReader


class BookSynthesizer:
    def __init__(
        self,
        book_path: str | Path,
    ):
        self.book = BookReader(book_path)
        self.book_title = self.book.get_book_title()
        self.cover_path = self.book.save_book_cover_image()
        self.language = self.book.get_language()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self._initialize_metadata()

    def _initialize_metadata(self):
        with open(self.audio_book_path / "chapters.txt", "w") as f:
            f.write(";FFMETADATA1\n")
            f.write("major_brand=M4A\n")
            f.write("minor_version=512\n")
            f.write("compatible_brands=M4A isomiso2\n")
            f.write("encoder=Lavf61.7.100\n")

    def synthesize(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        output_path: str | Path = "/data/output/",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.model = TTS(model_name=model_name)
        self.device = device
        self.model.to(self.device)
        self.audio_book_path = Path(f"{output_path}/{self.book_title}")
        self.audio_book_path.mkdir(parents=True, exist_ok=True)
        chapter_id = 1
        chapter_start = 0
        for chapter in self.book.read_chapters():
            chapter_start = self._process_chapter(
                chapter_id,
                chapter.sentences,
                chapter_start,
            )
            chapter_id += 1

    def _process_chapter(self, chapter_id, chapter, chapter_start):
        if (
            len(chapter) < 1000
            or Path(f"{self.audiobook_path}/chapter_{chapter_id}.txt").exists()
        ):
            if not Path(f"{self.audiobook_path}/chapter_{chapter_id}.txt").exists():
                duration = self._get_wav_duration(
                    f"{self.audiobook_path}/chapter_{chapter_id}.wav"
                )
                chapter_start += int(duration * 1000)
            return chapter_start
        self._chapter_to_speech(chapter.sentences, chapter_id)
        title = chapter.title
        duration = self._get_wav_duration(
            f"{self.audiobook_path}/chapter_{chapter_id}.wav"
        )
        chapter_start = self._log_on_chapter_file(
            f"{self.audiobook_path}/chapter_{chapter_id}.txt",
            title,
            chapter_start,
            duration,
        )
        return chapter_start

    def _log_on_chapter_file(self, title: str, start: int, duration: float) -> int:
        end = start + int(duration * 1000)
        with open(self.audio_book_path / "chapters.txt", "a") as f:
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start}\n")
            f.write(f"END={end}\n")
            f.write(f"TITLE={title}\n")
            f.write("\n")
        return end

    def _chapter_to_speech(
        self,
        sentences: list[str],
        chapter_number: int,
        audiobook_path: str | Path,
        language: str,
    ) -> None:
        sentence_to_speech(
            model=self.model,
            sentence=sentences[0],
            file_path=f"{audiobook_path}/chapter_{chapter_number}.wav",
            language=language,
        )
        combined_audio = AudioSegment.from_wav(
            f"{audiobook_path}/chapter_{chapter_number}.wav"
        )
        for sentence in sentences[1:]:
            sentence_to_speech(
                self.model,
                file_path=f"{self.tmp_dir}/speech.wav",
                sentence=sentence,
                language=language,
            )
            audio = AudioSegment.from_wav(f"{self.tmp_dir}/speech.wav")
            combined_audio += audio
            combined_audio.export(
                f"{audiobook_path}/chapter_{chapter_number}.wav", format="wav"
            )

    def _create_m4b(self, chapter_files, filename, cover_image_path: str | None = None):
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
            cover_image_file = tempfile.NamedTemporaryFile("wb")
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
                f"{self.audio_book_path}/chapters.txt",
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

    def _get_wav_duration(self, file_path):
        # Open the .wav file
        with wave.open(file_path, "rb") as wav_file:
            # Get the number of frames
            frames = wav_file.getnframes()
            # Get the frame rate (samples per second)
            frame_rate = wav_file.getframerate()
            # Calculate the duration in seconds
            duration = frames / float(frame_rate)
            return duration


def sentence_to_speech(
    model: TTS,
    sentence: str,
    file_path: str,
    language: str = "es",
    speaker: str | Path = "data/Jennifer_16khz.wav",
) -> None:
    if Path(file_path).parent.is_dir() is False:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        model.tts_to_file(
            text=sentence,
            file_path=file_path,
            language=language,
            speaker_wav=speaker,
        )
    except Exception as e:
        error_message = "Error: " + str(e)
        model.tts_to_file(
            text=error_message,
            file_path=file_path,
            language=language,
            speaker_wav=speaker,
        )
