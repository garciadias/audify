# %%
import sys
from pathlib import Path

import PyPDF2

from audify.utils import clean_text


class PdfReader:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"PDF file not found at {path}")

        # Extract text from PDF
        self.text = self._extract_text()

        # Clean the extracted text
        self.cleaned_text = clean_text(self.text)

    def _extract_text(self):
        """Extract text from a PDF file."""
        with open(self.path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def get_cleaned_text(self) -> str:
        """Get the cleaned text from the PDF."""
        return self.cleaned_text

    def save_cleaned_text(self, filename: str | Path):
        """Save the cleaned text to a file.

        Args:
            filename: Name of the file to save.
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.cleaned_text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pdf_reader.py <path_to_pdf>")
        sys.exit(1)

    reader = PdfReader(sys.argv[1])
    cleaned_text = reader.get_cleaned_text()

    # Print first 200 characters of cleaned text to verify
    print(cleaned_text[:200])

    # Optionally save the cleaned text to a file
    reader.save_cleaned_text("cleaned_article.txt")
