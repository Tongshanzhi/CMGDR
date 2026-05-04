"""Extract text embeddings from item title+description and aggregated reviews."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Clear broken proxy
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_text_extract(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root)
    cfg = _load_config(Path(config_path) if config_path else root / "config" / "text.yaml")
    up = cfg["upstream"]
    text_cfg = cfg.get("text", {})

    data_root = Path(up["data_root"])
    model_name = text_cfg.get("model_name", "all-MiniLM-L6-v2")
    batch_size = text_cfg.get("batch_size", 256)
    max_review_len = text_cfg.get("max_review_len", 256)
    max_reviews_per_item = text_cfg.get("max_reviews_per_item", 5)
    output_path = root / text_cfg.get("output_path", "data/processed/item_text_embeddings.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load item info
    with open(data_root / up["item_id_map"], encoding="utf-8") as f:
        asin_to_idx: dict[str, int] = json.load(f)
    n_items = len(asin_to_idx)
    idx_to_asin = {v: k for k, v in asin_to_idx.items()}
    print(f"Total items: {n_items}")

    # Load item metadata (title + description)
    items = pd.read_parquet(data_root / up["item_parquet"])
    items_by_asin = {}
    for _, row in items.iterrows():
        asin = row.get("asin", "")
        title = str(row.get("title", "") or "")
        desc = str(row.get("description", "") or "")
        items_by_asin[asin] = (title, desc)

    # Load reviews and aggregate per item (train split only to avoid leakage)
    interactions = pd.read_parquet(data_root / "processed/interactions.parquet")
    train_inter = interactions[interactions["split"] == "train"].copy()
    train_inter["reviewText"] = train_inter["reviewText"].fillna("").astype(str)

    # Aggregate: for each item, take up to max_reviews_per_item reviews, truncated
    review_agg: dict[int, str] = {}
    for item_idx in range(n_items):
        asin = idx_to_asin.get(item_idx, "")
        item_reviews = train_inter[train_inter["item_idx"] == item_idx]["reviewText"].tolist()
        # Take first N reviews, truncate each
        selected = [r[:max_review_len] for r in item_reviews[:max_reviews_per_item] if len(r) > 0]
        review_agg[item_idx] = " [SEP] ".join(selected)

    # Build text for each item: "Title: ... Description: ... Reviews: ..."
    texts = []
    for item_idx in range(n_items):
        asin = idx_to_asin.get(item_idx, "")
        title, desc = items_by_asin.get(asin, ("", ""))
        reviews = review_agg.get(item_idx, "")

        parts = []
        if title:
            parts.append(title)
        if desc:
            parts.append(desc[:256])
        if reviews:
            parts.append(reviews)

        text = " . ".join(parts) if parts else "unknown product"
        texts.append(text)

    print(f"Built {len(texts)} text inputs")
    print(f"Sample[0]: {texts[0][:150]}...")
    print(f"Sample[100]: {texts[100][:150]}...")

    # Encode with SentenceTransformer
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    print(f"Encoding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    print(f"Embeddings shape: {embeddings.shape}")
    np.save(output_path, embeddings)
    print(f"Saved: {output_path}")

    summary = {
        "n_items": n_items,
        "text_dim": int(embeddings.shape[1]),
        "model_name": model_name,
        "output_path": str(output_path),
        "n_with_title": sum(1 for t in texts if len(t) > 20),
        "n_empty": sum(1 for t in texts if t == "unknown product"),
    }
    return summary
