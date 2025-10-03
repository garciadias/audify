# %%
from pathlib import Path

import PyPDF2

from audify.domain.reader import Reader
from audify.utils.text import clean_text


class PdfReader(Reader):
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"PDF file not found at {path}")

        # Extract text from PDF
        self.text = self.read()

        # Clean the extracted text
        self.cleaned_text = clean_text(self.text)

    def read(self):
        """Extract text from a PDF file."""
        with open(self.path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def save_cleaned_text(self, file_name: str | Path):
        """Save the cleaned text to a file.

        Args:
            file_name: Name of the file to save.
        """
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(self.cleaned_text)
