# Downloads the registered sources, extracts their text and chunks it at headings. Embedding is a separate step.
from __future__ import annotations
from tc3_vqa.paths import CORPUS

import argparse
from pathlib import Path

from tc3vlm.corpus.chunk import chunk_pages, save_chunks
from tc3vlm.corpus.crawl import extract_archives, fetch_source, update_manifest
from tc3vlm.corpus.pdf_text import extract_pages
from tc3vlm.corpus.sources import TIER1


def process_source_files(source_id: str, raw_dir: Path, processed_dir: Path) -> int:
    src_raw = Path(raw_dir) / source_id
    out_dir = Path(processed_dir) / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Include PDFs nested inside extracted zips
    extract_archives(src_raw)
    pdfs = sorted(src_raw.rglob("*.pdf"))
    if not pdfs:
        print(f"  [skip] {source_id}: no PDFs")
        return 0

    total_chunks = 0
    for pdf in pdfs:
        try:
            pages = extract_pages(pdf)
            chunks = chunk_pages(source_id, pages)
            out_json = out_dir / f"{pdf.stem}.chunks.json"
            save_chunks(chunks, out_json)
            total_chunks += len(chunks)
            print(f"  {pdf.name}: {len(pages)} pages → {len(chunks)} chunks")
        except Exception as e:
            print(f"  [WARN] {pdf.name}: {type(e).__name__} {e}")
    return total_chunks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path,
                   default=Path(CORPUS + "/raw"))
    p.add_argument("--processed-dir", type=Path,
                   default=Path(CORPUS + "/processed"))
    p.add_argument("--manifest", type=Path,
                   default=Path(CORPUS + "/manifest.json"))
    p.add_argument("--source-id", default=None,
                   help="Process a single source. If omitted, process all of Tier 1.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip downloads (use when PDFs are already in raw/).")
    args = p.parse_args()

    targets = [s.id for s in TIER1] if args.source_id is None else [args.source_id]

    for sid in targets:
        print(f"\n[{sid}]")
        if not args.skip_download:
            entry = fetch_source(sid, args.raw_dir)
            update_manifest(args.manifest, entry)
            print(f"  download status: {entry['status']}")

        n = process_source_files(sid, args.raw_dir, args.processed_dir)
        if n:
            print(f"  total chunks: {n}")


if __name__ == "__main__":
    main()
