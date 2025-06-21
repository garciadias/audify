import tempfile
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from audify.pdf_read import PdfReader

MODULE_PATH = Path(__file__).resolve().parents[2]


@pytest.fixture
def pdfs() -> list[str]:
    # PDF file in english
    tmp_folder = tempfile.tempdir
    path_en = f"{tmp_folder}/test_en.pdf"
    c = canvas.Canvas(path_en, pagesize=(8.27 * 72, 11.7 * 72))
    c.drawString(100, 700, "This is a test PDF content.")
    c.save()
    path_pt = f"{tmp_folder}/test_pt.pdf"
    c = canvas.Canvas(path_pt, pagesize=(8.27 * 72, 11.7 * 72))
    c.drawString(100, 700, "Este é um conteúdo de teste em PDF.")
    c.save()
    return [str(path_en), str(path_pt)]


@pytest.fixture
def readers(pdfs):

    return [PdfReader(file_path) for file_path in pdfs]
