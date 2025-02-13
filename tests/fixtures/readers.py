from pathlib import Path

from audify.pdf_read import PdfReader

MODULE_PATH = Path(__file__).resolve().parents[2]
TEST_FILE_NAMES = ["test1", "test2"]
READERS = [
    PdfReader(MODULE_PATH / "data" / f"{file_name}.pdf")
    for file_name in TEST_FILE_NAMES
]
