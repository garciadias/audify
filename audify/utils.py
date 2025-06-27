import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
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
    # Remove multiple punctuation marks
    cleaned = re.sub(r"([.,!?;:¿¡-])+", r"\1", cleaned)
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
        sentence_words = sentence.split()
        new_sentence = ""
        for word in sentence_words:
            if len(new_sentence) + len(word) > max_length:
                result.append(new_sentence.strip(" "))
                new_sentence = ""
            new_sentence += word + " "
        if new_sentence:
            result.append(new_sentence.strip(" "))
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
    # Parallelize the cleaning of the sentences
    with ThreadPoolExecutor() as executor:
        result = list(executor.map(clean_text, result))
    return result


def get_audio_duration(file_path: str) -> float:
    # Load the audio file
    try:
        audio = AudioSegment.from_file(file_path)
    except CouldntDecodeError:
        logging.error(f"Could not decode audio file: {file_path}")
        return 0.0
    # Calculate the duration in seconds
    duration = len(audio) / 1000.0
    return duration


def sentence_to_speech(
    sentence: str,
    model: TTS,
    output_dir: Path | str = ("/tmp/"),
    language: str = "en",
    speaker: str | Path = "data/Jennifer_16khz.wav",
    file_name: str = "speech.wav",
) -> None:
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if Path(output_dir).parent.is_dir() is False:
        Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    try:
        if model.is_multi_lingual:
            model.tts_to_file(
                text=sentence,
                file_path=output_dir / file_name,
                language=language,
                speaker_wav=speaker,
                speed=1.15,
            )
        else:
            model.tts_to_file(
                text=sentence,
                file_path=output_dir / file_name,
                speaker_wav=speaker,
                speed=1.15,
            )
    except KeyError as e:
        error_message = "Error: " + str(e)
        if model.is_multi_lingual:
            model.tts_to_file(
                text=error_message,
                file_path=output_dir / file_name,
                language=language,
                speaker_wav=speaker,
                speed=1.15,
            )
        else:
            model.tts_to_file(
                text=error_message,
                file_path=output_dir / file_name,
                speaker_wav=speaker,
                speed=1.15,
            )


def get_file_extension(file_path: str) -> str:
    return Path(file_path).suffix


def get_file_name_title(title: str) -> str:
    # Make title snake_case and remove special characters and spaces
    title = title.lower().replace(" ", "_")
    # replace multiple underscores with a single one
    title = re.sub(r"_+", "_", title)
    # Remove leading and trailing underscores
    title = title.strip("_")
    # replace letter with accents using regex for simple letters
    title = re.sub(r"[àáâãäå]", "a", title)
    title = re.sub(r"[èéêë]", "e", title)
    title = re.sub(r"[ìíîï]", "i", title)
    title = re.sub(r"[òóôõö]", "o", title)
    title = re.sub(r"[ùúûü]", "u", title)
    title = re.sub(r"[ñ]", "n", title)
    title = re.sub(r"[ç]", "c", title)
    # Remove special characters
    title = re.sub(r"[^a-z0-9_]", "", title)  # Remove special characters
    # Remove leading and trailing underscores
    title = title.strip("_")
    return title
