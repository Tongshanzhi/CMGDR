"""Batch download product images from imUrl, keyed by item_idx."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Clear broken proxy env vars that interfere with requests
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_key, None)

import pandas as pd
import requests
import yaml
from tqdm import tqdm


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_download_list(
    data_root: Path, item_parquet_rel: str, item_id_map_rel: str
) -> list[tuple[int, str]]:
    """Return [(item_idx, url), ...] for items that have a non-empty imUrl."""
    items = pd.read_parquet(data_root / item_parquet_rel)
    with open(data_root / item_id_map_rel, encoding="utf-8") as f:
        asin_to_idx: dict[str, int] = json.load(f)

    merged = items[["asin", "imUrl"]].copy()
    merged["item_idx"] = merged["asin"].map(asin_to_idx)
    merged = merged.dropna(subset=["item_idx"])
    merged["item_idx"] = merged["item_idx"].astype(int)
    merged = merged[merged["imUrl"].str.len() > 0]
    return list(zip(merged["item_idx"].tolist(), merged["imUrl"].tolist()))


def _download_one(
    item_idx: int,
    url: str,
    output_dir: Path,
    timeout: int,
    max_retries: int,
    retry_delay: float,
    min_bytes: int,
    user_agent: str,
) -> dict[str, Any]:
    dest = output_dir / f"{item_idx}.jpg"
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return {"item_idx": item_idx, "status": "skipped", "path": str(dest)}

    headers = {"User-Agent": user_agent}
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            if tmp.stat().st_size < min_bytes:
                tmp.unlink(missing_ok=True)
                last_error = f"too small ({tmp.stat().st_size} B)"
                continue
            tmp.replace(dest)
            return {"item_idx": item_idx, "status": "ok", "path": str(dest)}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

    return {"item_idx": item_idx, "status": "failed", "error": last_error, "url": url}


def run_download(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root)
    cfg = _load_config(Path(config_path) if config_path else root / "config" / "visual.yaml")
    up = cfg["upstream"]
    dl = cfg["download"]

    data_root = Path(up["data_root"])
    output_dir = root / dl["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load item_id_map to know total n_items
    with open(data_root / up["item_id_map"], encoding="utf-8") as f:
        n_items = len(json.load(f))

    download_list = _build_download_list(data_root, up["item_parquet"], up["item_id_map"])
    all_item_idxs = set(range(n_items))
    has_url_idxs = {idx for idx, _ in download_list}
    no_url_idxs = sorted(all_item_idxs - has_url_idxs)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=dl["max_workers"]) as pool:
        futures = {
            pool.submit(
                _download_one,
                idx, url, output_dir,
                dl["timeout_sec"], dl["max_retries"],
                dl["retry_delay_sec"], dl["min_image_bytes"],
                dl["user_agent"],
            ): idx
            for idx, url in download_list
        }
        with tqdm(total=len(futures), desc="Downloading images", unit="img") as pbar:
            for fut in as_completed(futures):
                results.append(fut.result())
                pbar.update(1)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]

    manifest = {
        "n_items": n_items,
        "n_with_url": len(download_list),
        "n_no_url": len(no_url_idxs),
        "no_url_item_idxs": no_url_idxs,
        "n_downloaded": len(ok),
        "n_skipped_exist": len(skipped),
        "n_failed": len(failed),
        "failed": [{"item_idx": r["item_idx"], "url": r.get("url", ""), "error": r.get("error", "")} for r in failed],
    }

    manifest_path = root / dl["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Download complete: {len(ok)} new + {len(skipped)} existing, "
          f"{len(failed)} failed, {len(no_url_idxs)} no URL")
    return manifest
