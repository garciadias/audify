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
def main(epub_path: str):
    book_synthesizer = EpubSynthesizer(epub_path)
    book_synthesizer.synthesize()


if __name__ == "__main__":
    main()
