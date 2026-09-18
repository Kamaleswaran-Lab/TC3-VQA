# Retrieves arXiv preprints for the auxiliary literature set through the export API and stores title and abstract as
# chunks.
from __future__ import annotations
from tccc_vqa.paths import CORPUS

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import requests

from .chunk import Chunk

ARXIV_SOURCE_ID = "arxiv_tccc_v1"
ARXIV_API = "http://export.arxiv.org/api/query"
_UA = {"User-Agent": "TCCC-VQA-corpus-builder/1.0 (research)"}

QUERIES: list[tuple[str, str]] = [
    ("tccc_direct",
     'all:"tactical combat casualty care" OR all:"TCCC"'),
    ("combat_medical_ai",
     '(all:"combat medical" OR all:"battlefield medical" OR all:"military medical") AND '
     '(all:"machine learning" OR all:"deep learning" OR all:"vision language" OR all:"large language model")'),
    ("medical_vlm_general",
     '(all:"vision-language model" OR all:"vision language model" OR all:"VLM") AND '
     '(all:"medical" OR all:"clinical" OR all:"radiology" OR all:"trauma")'),
    ("medical_rag",
     '(all:"retrieval augmented" OR all:"retrieval-augmented" OR all:"RAG") AND '
     '(all:"medical" OR all:"clinical" OR all:"healthcare")'),
    ("medical_video_understanding",
     '(all:"medical video" OR all:"surgical video" OR all:"clinical video") AND '
     '(all:"foundation model" OR all:"vision language" OR all:"video instruction")'),
    ("triage_ai",
     '(all:"triage" OR all:"casualty care") AND (all:"AI" OR all:"machine learning")'),
    ("austere_medicine",
     'all:"prolonged field care" OR all:"prolonged casualty care" OR all:"austere medicine"'),
]


def query_arxiv(query: str, max_results: int = 200, sleep: float = 3.0) -> list[dict]:
    """arXiv API expects ~3s between requests. Returns list of dicts (title, summary, authors, id, date)."""
    import xml.etree.ElementTree as ET

    params = {
        "search_query": query,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API, params=params, headers=_UA, timeout=60)
    r.raise_for_status()
    time.sleep(sleep)
    ns = {"a": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(r.text)
    out: list[dict] = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
        arxiv_id = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        authors = [a.findtext("a:name", default="", namespaces=ns)
                   for a in entry.findall("a:author", ns)]
        cats = [c.attrib.get("term", "") for c in entry.findall("a:category", ns)]
        # Skip the dummy entry sometimes returned for no results
        if not title or title.lower().startswith("error"):
            continue
        if len(summary) < 100:
            continue
        out.append({
            "id": arxiv_id, "title": title, "summary": summary,
            "authors": authors, "published": published,
            "categories": cats,
        })
    return out


def crawl_arxiv(raw_dir: Path, processed_dir: Path, max_per_query: int = 200) -> int:
    raw_dir = Path(raw_dir) / ARXIV_SOURCE_ID
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(processed_dir) / ARXIV_SOURCE_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    by_id: dict[str, dict] = {}
    per_query: dict[str, list[str]] = {}
    for label, q in QUERIES:
        try:
            results = query_arxiv(q, max_results=max_per_query)
        except Exception as e:
            print(f"  {label:30s} ERROR {e}")
            continue
        per_query[label] = [r["id"] for r in results]
        for r in results:
            by_id.setdefault(r["id"], r)
        print(f"  {label:30s} {len(results):5d} results  (total unique: {len(by_id)})")

    (raw_dir / "records.json").write_text(json.dumps(list(by_id.values()), indent=2, ensure_ascii=False))
    (raw_dir / "per_query_ids.json").write_text(json.dumps(per_query, indent=2))

    chunks: list[Chunk] = []
    for idx, rec in enumerate(by_id.values()):
        # Skip the v1 TC3-VLM paper itself if it appears
        if "TC3-VLM" in rec["title"] or "TC3-VLM" in rec["summary"][:200]:
            continue
        section = re.sub(r"\W+", "_", (rec["categories"][0] if rec["categories"] else "arxiv"))[:40]
        clean_id = rec["id"].rsplit("/", 1)[-1]
        cid = f"{ARXIV_SOURCE_ID}:{section}:arxiv_{clean_id}"
        text_parts = [
            f"Title: {rec['title']}",
            f"Authors: {', '.join(rec['authors'][:6])}{' et al.' if len(rec['authors']) > 6 else ''}",
            f"arXiv: {clean_id}  ({rec['published'][:10]})",
            f"Categories: {', '.join(rec['categories'])}",
            "",
            rec["summary"],
        ]
        chunks.append(Chunk(
            citation_id=cid, source_id=ARXIV_SOURCE_ID,
            text="\n".join(text_parts),
            page_start=1, page_end=1,
            section_path=section, chunk_idx=idx,
            extra={"arxiv_id": clean_id, "categories": rec["categories"]},
        ))

    out_path = out_dir / "arxiv.chunks.json"
    out_path.write_text(json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False))
    return len(chunks)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path(CORPUS + "/raw"))
    p.add_argument("--processed-dir", type=Path,
                   default=Path(CORPUS + "/processed"))
    p.add_argument("--max-per-query", type=int, default=200)
    args = p.parse_args()
    n = crawl_arxiv(args.raw_dir, args.processed_dir, args.max_per_query)
    print(f"arXiv chunks written: {n}")
