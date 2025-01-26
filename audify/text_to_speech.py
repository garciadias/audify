# %%
import subprocess
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from pydub import AudioSegment
from TTS.api import TTS

from audify import ebook_read

MODULE_PATH = Path(__file__).parents[1]

# Get device
device = "cuda" if torch.cuda.is_available() else False
# %%

LOADED_MODEL = TTS("tts_models/es/mai/tacotron2-DDC", gpu=device)
# %%


def sentence_to_speech(
    sentence: str,
    file_path: str = f"{MODULE_PATH}/data/output/speech.wav",
) -> None:
    if Path(file_path).parent.is_dir() is False:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        LOADED_MODEL.tts_to_file(
            text=sentence,
            file_path=file_path,
        )
    except Exception as e:
        error_message = "Error: " + str(e)
        LOADED_MODEL.tts_to_file(
            text=error_message,
            file_path=file_path,
        )


def process_sentence(sentence):
    sentence_to_speech(sentence=sentence)
    audio = AudioSegment.from_wav(f"{MODULE_PATH}/data/output/speech.wav")
    return audio


def synthesize_chapter(chapter: str, chapter_number: int) -> None:
    chapter_txt = ebook_read.extract_text_from_epub_chapter(chapter)
    sentences = ebook_read.break_text_into_sentences(chapter_txt)
    sentence_to_speech(
        sentence=sentences[0],
        file_path=f"{MODULE_PATH}/data/output/chapter_{chapter_number}.wav",
    )
    combined_audio = AudioSegment.from_wav(
        f"{MODULE_PATH}/data/output/chapter_{chapter_number}.wav"
    )
    for sentence in sentences[1:]:
        sentence_to_speech(sentence=sentence)
        audio = AudioSegment.from_wav(f"{MODULE_PATH}/data/output/speech.wav")
        combined_audio += audio
        combined_audio.export(
            f"{MODULE_PATH}/data/output/chapter_{chapter_number}.wav", format="wav"
        )


def create_m4b(chapter_files, filename, cover_image_path: str = None):
    tmp_filename = filename.replace(".epub", ".tmp.mp4")
    if not Path(tmp_filename).exists():
        combined_audio = AudioSegment.empty()
        for wav_file in chapter_files:
            audio = AudioSegment.from_wav(wav_file)
            combined_audio += audio
        print("Converting to M4b...")
        combined_audio.export(tmp_filename, format="mp4", codec="aac", bitrate="64k")
    final_filename = filename.replace(".epub", ".m4b")
    print("Creating M4B file...")

    with open(cover_image_path, "rb") as f:
        cover_image = f.read()

    if cover_image:
        cover_image_file = NamedTemporaryFile("wb")
        cover_image_file.write(cover_image)
        cover_image_args = ["-i", cover_image_file.name, "-map", "0:a", "-map", "2:v"]
    else:
        cover_image_args = []
    cover_image_path = Path(cover_image_path)
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


def log_on_chapter_file(
    chapter_file_path: Path | str, title: str, start: int, duration: float
) -> int:
    end = start + int(duration * 1000)
    if isinstance(chapter_file_path, str):
        chapter_file_path = Path(chapter_file_path)
    with open(chapter_file_path.parent / "chapters.txt", "a") as f:
        f.write(f"[CHAPTER]{chapter_file_path.stem}\n")
        f.write(f"TITLE={title}\n")
        f.write("TIMEBASE=1/1000\n")
        f.write(f"START={start}\n")
        f.write(f"END={end}\n")
        f.write("\n")
    return end


def process_chapter(i, chapter, chapter_start):
    if len(chapter) < 1000:
        return chapter_start
    print(f"Synthesizing chapter: {i}")
    synthesize_chapter(chapter, i)
    title = f"Chapter {i}: {ebook_read.get_chapter_title(chapter)}"
    duration = get_wav_duration(f"{MODULE_PATH}/data/output/chapter_{i}.wav")
    chapter_start = log_on_chapter_file(
        f"{MODULE_PATH}/data/output/chapter_{i}.txt", title, chapter_start, duration
    )
    return chapter_start
