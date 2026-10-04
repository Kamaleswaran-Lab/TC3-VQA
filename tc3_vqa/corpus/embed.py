# Builds the BGE-large embeddings and the FAISS index from the chunks. Run once; the index is read by the retrieval steps.
from __future__ import annotations

import glob
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def load_all_chunks(processed_dir: Path, min_chars: int = 80) -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(glob.glob(str(Path(processed_dir) / "*" / "*.chunks.json"))):
        for c in json.loads(Path(path).read_text()):
            text = (c.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            c["text"] = text
            chunks.append(c)
    return chunks


def dedup_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate by sha256(text). Needed because the JTS Top-10 zip overlaps with A-H/I-Z."""
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        h = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        c["text_sha256"] = h
        out.append(c)
    return out


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_corpus_index(
    chunks: list[dict],
    output_dir: Path,
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str | None = None,
    batch_size: int = 64,
) -> dict:
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [_clean_text(c["text"]) for c in chunks]

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    np.save(output_dir / "embeddings.npy", embeddings)

    with open(output_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "faiss.index"))

    meta = {
        "model_name": model_name,
        "embedding_dim": dim,
        "n_chunks": len(chunks),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta
