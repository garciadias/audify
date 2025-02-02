from pathlib import Path

import click

from audify.text_to_speech import EpubSynthesizer

MODULE_PATH = Path(__file__).resolve().parents[1]


@click.command()
# Path to the epub file to be synthesized is required as default parameter
@click.argument(
    "epub_path",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--language",
    "-l",
    type=click.Choice(["en", "es", "pt"], case_sensitive=False),
    default="en",
    help="Language of the synthesized audiobook.",
)
@click.option(
    "--voice",
    "-v",
    type=str,
    default="data/Jennifer_16khz.wav",
    help="Path to the speaker's voice.",
    )
def main(epub_path: str, language: str, voice: str):
    book_synthesizer = EpubSynthesizer(epub_path, language=language, speaker=voice)
    book_synthesizer.synthesize()


if __name__ == "__main__":
    main()
