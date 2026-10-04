# Splits page text into retrieval chunks at detected headings, keeping the citation id, section path and page range.
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .pdf_text import PageText


@dataclass
class Chunk:
    citation_id: str          # source_id:section_path:idx
    source_id: str
    text: str
    page_start: int
    page_end: int
    section_path: str = ""
    chunk_idx: int = 0
    extra: dict = field(default_factory=dict)


# Recognize headings by ALL CAPS lines, numbered prefixes, and "Chapter X" / "Section X" patterns.
# Case-sensitive on the keyword so that prose like "section of the TCCC Guidelines" is rejected.
# Numbered prefix requires at least one dot (e.g., "3.2.1") so calendar dates ("01 May 2023") don't match.
_HEADING_PATTERNS = [
    re.compile(r"^\s*(SECTION|CHAPTER|APPENDIX|PART)\s+[A-Z0-9\.\-]+\b"),
    re.compile(r"^\s*\d+(\.\d+){1,3}\s+[A-Z][A-Za-z0-9\s\-,&/]{2,}$"),
    re.compile(r"^\s*[A-Z][A-Z\s\-,&/]{4,}$"),  # ALL CAPS line
]

# Patterns that look like headings but are actually dates, page markers, or section-prose. Reject first.
_HEADING_NEGATIVE_PATTERNS = [
    # Calendar dates: "25 January 2024", "01 May 2023"
    re.compile(
        r"^\s*\d{1,2}\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{2,4}\s*$",
        re.IGNORECASE,
    ),
    # ALL-CAPS lines that are clearly running headers / page markers, e.g., "ATP 4-02.11" alone
    re.compile(r"^\s*[A-Z]{2,5}\s+\d+(-\d+(\.\d+)?)?\s*$"),
    # Page footers / numbers only
    re.compile(r"^\s*\d+\s*$"),
    # "section of the …" prose — start with lowercase "section"
    re.compile(r"^\s*section of\b", re.IGNORECASE),
]


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    if any(p.match(line) for p in _HEADING_NEGATIVE_PATTERNS):
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _approx_tokens(text: str) -> int:
    # Fast approximation; precise tokenization happens at the embedding stage.
    return max(1, len(text) // 4)


def chunk_pages(
    source_id: str,
    pages: list[PageText],
    target_tokens: int = 500,
    max_tokens: int = 800,
) -> list[Chunk]:
    """Build chunks using heading detection and a token budget.
    A heading starts a new chunk; sections exceeding max_tokens are force-split."""
    chunks: list[Chunk] = []
    cur_text: list[str] = []
    cur_section = ""
    cur_page_start: int | None = None
    cur_page_end: int | None = None
    cur_tokens = 0

    def flush(text_parts: list[str], section: str, p_start: int, p_end: int) -> None:
        text = "\n".join(text_parts).strip()
        if not text:
            return
        idx = len(chunks)
        cid = f"{source_id}:{section or 'root'}:{idx:04d}"
        chunks.append(Chunk(
            citation_id=cid, source_id=source_id, text=text,
            page_start=p_start, page_end=p_end,
            section_path=section, chunk_idx=idx,
        ))

    for page in pages:
        for line in page.text.splitlines():
            if _looks_like_heading(line):
                flush(cur_text, cur_section,
                      cur_page_start or page.page_num,
                      cur_page_end or page.page_num)
                cur_text = []
                cur_section = line.strip()
                cur_page_start = page.page_num
                cur_page_end = page.page_num
                cur_tokens = 0
                continue

            line_tokens = _approx_tokens(line)
            if cur_tokens + line_tokens > max_tokens and cur_text:
                flush(cur_text, cur_section,
                      cur_page_start or page.page_num,
                      cur_page_end or page.page_num)
                cur_text = []
                cur_tokens = 0
                cur_page_start = page.page_num

            cur_text.append(line)
            cur_tokens += line_tokens
            cur_page_end = page.page_num

    flush(cur_text, cur_section, cur_page_start or 1, cur_page_end or 1)
    return chunks


def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        [asdict(c) for c in chunks], indent=2, ensure_ascii=False,
    ))
