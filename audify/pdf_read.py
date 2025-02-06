# %%
import re
import sys
from pathlib import Path

import PyPDF2


class PdfReader:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"PDF file not found at {path}")

        # Extract text from PDF
        self.text = self._extract_text()

        # Clean the extracted text
        self.cleaned_text = self._clean_text(self.text)

    def _extract_text(self):
        """Extract text from a PDF file."""
        with open(self.path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def _clean_text(self, text: str) -> str:
        """Clean the extracted text by removing unwanted elements.

        Args:
            text: Raw text extracted from PDF.

        Returns:
            Cleaned text suitable for TTS.
        """
        # Remove LaTeX formatting and special characters
        cleaned = re.sub(r"\$(.*?)(\$)", "", text)  # Remove equations in $...$ notation
        cleaned = re.sub(
            r"\\(figure|table)[^\\]*", "", cleaned
        )  # Remove figure/table references
        cleaned = re.sub(r"%.*?\n?", "", cleaned)  # Remove comments
        cleaned = re.sub(r"\{[^}]+\}", "", cleaned)  # Remove curly braces
        cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)  # Remove other LaTeX commands

        # Remove citations (common formats)
        cleaned = re.sub(r"\[(.*?)\]", "", cleaned)  # Remove [author year] format
        cleaned = re.sub(r"\(.*?\)", "", cleaned)  # Remove (author year) format
        cleaned = re.sub(r"\[?\d+\]?", "", cleaned)  # Remove standalone numbers

        # Remove footnotes
        cleaned = re.sub(r"Footnote:\s*\d+", "", cleaned)

        # Remove section headers and numbering
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^###?\s*", "", cleaned, flags=re.MULTILINE)

        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

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
