import re
import wave
from pathlib import Path

from TTS.api import TTS


def clean_text(text: str) -> str:
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Replace @ with 'a' to avoid TTS errors
    cleaned = cleaned.replace("@", "a")
    # Remove multiple spaces
    cleaned = re.sub(r" +", " ", cleaned)
    # Remove leading and trailing spaces
    cleaned = cleaned.strip()
    # Remove spaces before punctuation, commas, brackets, quotes, and hyphens
    cleaned = re.sub(r" ([.,!?;:¿¡-])", r"\1", cleaned)
    # Removes multiple spaces, tabs, newlines, punctuation and brackets
    cleaned = re.sub(r"[\s\[\]{}()<>/\\#]", " ", cleaned)
    # Remove extra spaces
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned


def combine_small_sentences(sentences: list[str], min_length: int = 10) -> list[str]:
    result: list[str] = []
    for sentence in sentences:
        if len(sentence) < min_length:
            if result:
                result[-1] += " " + sentence
        else:
            result.append(sentence)
    return result


def break_too_long_sentences(sentences: list[str], max_length: int = 239) -> list[str]:
    result: list[str] = []
    for sentence in sentences:
        while len(sentence) > max_length:
            result.append(sentence[:max_length])
            sentence = sentence[max_length:]
        if sentence:
            result.append(sentence)
    return result


def break_text_into_sentences(
    text: str, max_length: int = 239, min_length: int = 10
) -> list[str]:
    # Split text into sentences using punctuation marks
    sentences = re.split(r"(?<=[.!?;:¿¡]) +", text)
    # Split long sentences into smaller ones to avoid TTS errors
    result = break_too_long_sentences(sentences, max_length - min_length)
    # Combine sentences that are too short with the previous one
    result = combine_small_sentences(result, min_length)
    return result


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


def sentence_to_speech(
    sentence: str, tmp_dir: Path, language: str, speaker: str | Path, model: TTS
) -> None:
    if Path(tmp_dir).parent.is_dir() is False:
        Path(tmp_dir).parent.mkdir(parents=True, exist_ok=True)
    try:
        model.tts_to_file(
            text=sentence,
            file_path=tmp_dir / "speech.wav",
            language=language,
            speaker_wav=speaker,
        )
    except Exception as e:
        error_message = "Error: " + str(e)
        model.tts_to_file(
            text=error_message,
            file_path=tmp_dir / "speech.wav",
            language=language,
            speaker_wav=speaker,
        )


def get_file_extension(file_path: str) -> str:
    return Path(file_path).suffix
