# Downloads the files of a registered source into raw/ and records them in the manifest with their sha256.
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from .sources import Source, get_source


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _filename_from_url(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        return fallback
    # Use fallback name when the URL path lacks a known extension (e.g. /api/v1/pdf/{uuid}/contents)
    if not any(name.lower().endswith(ext) for ext in (".pdf", ".zip", ".html", ".json", ".csv", ".txt")):
        return fallback
    return name


def _request_headers(url: str) -> dict:
    """Send a desktop-browser User-Agent + a same-origin Referer.
    Required by armypubs.army.mil and similar portals that 403 anonymous bot fetches."""
    parsed = urlparse(url)
    return {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36",
        "Referer": f"{parsed.scheme}://{parsed.netloc}/",
        "Accept": "*/*",
    }


def fetch_source(
    source_id: str,
    raw_dir: Path,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> dict:
    """Download every URL on `source` into raw_dir/<source_id>/.
    Returns a manifest entry dict; caller merges into the global manifest."""
    src = get_source(source_id)
    out_dir = Path(raw_dir) / src.id
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.fetch != "http":
        return {"source_id": src.id, "status": "skipped",
                "reason": f"fetch={src.fetch}, manual download required"}
    if not src.urls:
        return {"source_id": src.id, "status": "skipped",
                "reason": "urls empty; fill them in sources.py"}

    files: list[dict] = []
    for url in src.urls:
        target = out_dir / _filename_from_url(url, f"{src.id}.pdf")
        if target.exists() and target.stat().st_size > 0:
            files.append({"url": url, "path": str(target),
                          "sha256": _hash_file(target), "status": "cached"})
            continue

        ok = False
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                with requests.get(url, stream=True, timeout=timeout,
                                  headers=_request_headers(url)) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    with open(target, "wb") as f, tqdm(
                        total=total, unit="B", unit_scale=True,
                        desc=target.name, leave=False,
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
                ok = True
                break
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(backoff ** attempt)

        if ok:
            files.append({"url": url, "path": str(target),
                          "sha256": _hash_file(target), "status": "downloaded"})
        else:
            files.append({"url": url, "path": str(target),
                          "status": "failed", "error": str(last_err)})

    return {
        "source_id": src.id,
        "tier": src.tier,
        "policy": src.policy,
        "version": src.version,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "status": "ok" if all(f["status"] != "failed" for f in files) else "partial",
    }


def update_manifest(manifest_path: Path, entry: dict) -> None:
    manifest_path = Path(manifest_path)
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest[entry["source_id"]] = entry
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def extract_archives(source_dir: Path) -> list[Path]:
    """Extract every *.zip in source_dir in place, return paths of extracted PDFs.
    Skips a zip when a (zipname).extracted marker file already exists."""
    import zipfile
    source_dir = Path(source_dir)
    pdfs: list[Path] = []
    for zip_path in source_dir.glob("*.zip"):
        marker = zip_path.with_suffix(".extracted")
        if not marker.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(source_dir)
            marker.touch()
        # Collect every PDF, including those inside extracted folders (recursive)
        for pdf in source_dir.rglob("*.pdf"):
            pdfs.append(pdf)
    return pdfs
