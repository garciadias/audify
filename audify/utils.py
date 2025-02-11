import re
import wave
from pathlib import Path

from TTS.api import TTS


def break_text_into_sentences(text: str) -> list[str]:
    # Replace @ with 'a' to avoid TTS errors
    text = text.replace("@", "a")
    sentences = re.split(r"(?<=[.!?;:¿¡]) +", text)
    # Remove extra spaces
    sentences = [re.sub(r" +", " ", sentence) for sentence in sentences]
    # Remove leading and trailing spaces
    sentences = [sentence.strip() for sentence in sentences]
    # Remove empty sentences
    sentences = [sentence for sentence in sentences if sentence]
    # Split long sentences into smaller ones to avoid TTS errors
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        while len(sentence) > 239:
            result.append(sentence[:239])
            sentence = sentence[239:]
        if sentence:
            result.append(sentence)
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
