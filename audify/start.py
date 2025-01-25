from pathlib import Path

from pydub import AudioSegment

from audify import ebook_read, text_to_speech

MODULE_PATH = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    text = ebook_read.read_chapters(f"{MODULE_PATH}/data/test.epub")
    text = ebook_read.extract_text_from_epub_chapter(text[10])
    sentences = ebook_read.break_text_into_sentences(text)
    text_to_speech.sentence_to_speech(
        sentence=sentences[0], file_path=f"{MODULE_PATH}/data/output/chapter.wav"
    )
    combined_audio = AudioSegment.from_wav(f"{MODULE_PATH}/data/output/chapter.wav")
    for sentence in sentences[1:]:
        text_to_speech.sentence_to_speech(sentence=sentence)
        audio = AudioSegment.from_wav(f"{MODULE_PATH}/data/output/speech.wav")
        combined_audio += audio
        combined_audio.export(f"{MODULE_PATH}/data/output/chapter.wav", format="wav")
    print("Chapter audio generated.")
