# Extracts per-page text from a PDF with PyMuPDF.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PageText:
    page_num: int     # 1-based
    text: str


def extract_pages(pdf_path: Path) -> list[PageText]:
    import fitz  # PyMuPDF
    pages: list[PageText] = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append(PageText(page_num=i, text=text))
    return pages
