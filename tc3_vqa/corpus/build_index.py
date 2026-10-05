# Loads all processed chunks, removes duplicates and builds the BGE-large embeddings and the FAISS index.
from __future__ import annotations
from tc3_vqa.paths import CORPUS

import argparse
from pathlib import Path

from tc3_vqa.corpus.embed import build_corpus_index, dedup_chunks, load_all_chunks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=Path,
                   default=Path(CORPUS + "/processed"))
    p.add_argument("--index-dir", type=Path,
                   default=Path(CORPUS + "/index"))
    p.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--device", default=None,
                   help="None lets torch.cuda.is_available() pick automatically.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-chars", type=int, default=80)
    args = p.parse_args()

    print(f"Loading chunks from {args.processed_dir}/")
    raw = load_all_chunks(args.processed_dir, min_chars=args.min_chars)
    print(f"  raw chunks (after min_chars filter): {len(raw)}")

    dedup = dedup_chunks(raw)
    print(f"  after sha256 dedup: {len(dedup)} ({len(raw) - len(dedup)} duplicates removed)")

    print(f"\nBuilding index with {args.model} on {args.device}...")
    meta = build_corpus_index(
        dedup, args.index_dir,
        model_name=args.model, device=args.device, batch_size=args.batch_size,
    )
    print(f"\n[OK] index built")
    print(f"  dir: {args.index_dir}")
    print(f"  dim: {meta['embedding_dim']}")
    print(f"  n_chunks: {meta['n_chunks']}")


if __name__ == "__main__":
    main()
