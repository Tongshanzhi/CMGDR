"""Download Amazon category reviews (5-core) and metadata.

Dataset: Amazon Reviews 2023 — see https://amazon-reviews-2023.github.io/
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml


def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _category_files(category: str) -> tuple[str, str]:
    reviews = f"reviews_{category}_5.json.gz"
    meta = f"meta_{category}.json.gz"
    return reviews, meta


def download_file(url: str, dest: Path, chunk_size: int, timeout: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def ensure_downloaded(
    project_root: Path,
    config_path: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Returns paths to local gz files: reviews_gz, meta_gz.
    """
    root = project_root
    cfg_path = config_path or root / "config" / "default.yaml"
    cfg = _load_config(cfg_path)
    ds = cfg["dataset"]
    paths = cfg["paths"]
    dl = cfg["download"]

    category = ds["category"]
    base = ds["base_url"].rstrip("/") + "/"
    raw_dir = root / paths["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    rev_name, meta_name = _category_files(category)
    rev_url = urljoin(base, rev_name)
    meta_url = urljoin(base, meta_name)

    out_rev = raw_dir / rev_name
    out_meta = raw_dir / meta_name

    if force or not out_rev.exists():
        download_file(rev_url, out_rev, dl["chunk_size"], dl["timeout_sec"])
    if force or not out_meta.exists():
        download_file(meta_url, out_meta, dl["chunk_size"], dl["timeout_sec"])

    return {"reviews_gz": out_rev, "meta_gz": out_meta, "category": category}


def gunzip_if_needed(gz_path: Path, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return dest
