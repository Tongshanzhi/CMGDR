"""Parse Amazon JSONL, k-core filtering, temporal split, multimodal side table."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_reviews_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Loading reviews", unit="lines"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append(
                {
                    "reviewerID": o.get("reviewerID"),
                    "asin": o.get("asin"),
                    "overall": float(o.get("overall", 0)),
                    "unixReviewTime": int(o.get("unixReviewTime", 0)),
                    "reviewText": o.get("reviewText") or "",
                    "summary": o.get("summary") or "",
                }
            )
    return pd.DataFrame(rows)


def load_meta_jsonl(path: Path) -> pd.DataFrame:
    """Amazon meta 文件为 Python 字面量（单引号），非标准 JSON。

    数据集：Amazon Reviews 2023 — https://amazon-reviews-2023.github.io/
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Loading meta", unit="lines"):
            line = line.strip()
            if not line:
                continue
            try:
                o = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(o, dict):
                continue
            rel = o.get("related") if isinstance(o.get("related"), dict) else {}
            also_bought = list(rel.get("also_bought") or rel.get("also_buy") or [])
            rows.append(
                {
                    "asin": o.get("asin"),
                    "title": (o.get("title") or "") if o.get("title") is not None else "",
                    "description": (o.get("description") or "") if o.get("description") is not None else "",
                    "imUrl": (o.get("imUrl") or "") if o.get("imUrl") is not None else "",
                    "brand": (o.get("brand") or "") if o.get("brand") is not None else "",
                    "categories": o.get("categories"),
                    "also_buy": also_bought,
                    "also_viewed": list(rel.get("also_viewed") or []),
                    "bought_together": list(rel.get("bought_together") or []),
                }
            )
    return pd.DataFrame(rows)


def k_core_filter(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    k_u: int,
    k_i: int,
) -> pd.DataFrame:
    """Iterative k-core on users and items."""
    out = df
    while True:
        u_cnt = out.groupby(user_col).size()
        keep_u = set(u_cnt[u_cnt >= k_u].index)
        out = out[out[user_col].isin(keep_u)]
        i_cnt = out.groupby(item_col).size()
        keep_i = set(i_cnt[i_cnt >= k_i].index)
        out = out[out[item_col].isin(keep_i)]
        if len(out) == len(df):
            break
        df = out
    return out.reset_index(drop=True)


def reindex_ids(df: pd.DataFrame, user_col: str, item_col: str) -> tuple[pd.DataFrame, dict, dict]:
    u_ids = df[user_col].unique()
    i_ids = df[item_col].unique()
    u_map = {u: i for i, u in enumerate(u_ids)}
    i_map = {it: i for i, it in enumerate(i_ids)}
    out = df.copy()
    out["user_idx"] = out[user_col].map(u_map)
    out["item_idx"] = out[item_col].map(i_map)
    return out, u_map, i_map


def temporal_split(
    df: pd.DataFrame,
    time_col: str,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    split = ["train"] * n_train + ["valid"] * n_val + ["test"] * n_test
    # 边界取整可能导致长度不一致
    if len(split) < n:
        split += ["train"] * (n - len(split))
    elif len(split) > n:
        split = split[:n]
    out = df.copy()
    out["split"] = split
    return out


def run_preprocess(
    project_root: Path,
    config_path: Path | None = None,
    reviews_jsonl: Path | None = None,
    meta_jsonl: Path | None = None,
) -> dict:
    cfg_path = config_path or project_root / "config" / "default.yaml"
    cfg = _load_config(cfg_path)
    paths = cfg["paths"]
    pre = cfg["preprocess"]
    sp = cfg["split"]

    proc_dir = project_root / paths["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    if reviews_jsonl is None or meta_jsonl is None:
        raise ValueError("reviews_jsonl and meta_jsonl required (gunzip raw first)")

    rev = load_reviews_jsonl(reviews_jsonl)
    meta = load_meta_jsonl(meta_jsonl)

    rev = rev.dropna(subset=["reviewerID", "asin"])
    rev = rev[(rev["overall"] >= pre["min_rating"]) & (rev["overall"] <= 5)]
    if pre.get("require_review_text"):
        rev = rev[rev["reviewText"].str.len() > 0]

    if pre.get("max_reviews_per_user", 0) and pre["max_reviews_per_user"] > 0:
        rev = (
            rev.sort_values("unixReviewTime")
            .groupby("reviewerID", group_keys=False)
            .head(pre["max_reviews_per_user"])
        )

    rev = k_core_filter(
        rev,
        "reviewerID",
        "asin",
        pre["k_core_users"],
        pre["k_core_items"],
    )

    rev, user_map, item_map = reindex_ids(rev, "reviewerID", "asin")
    rev = temporal_split(rev, sp["time_column"], sp["val_ratio"], sp["test_ratio"])

    meta_sub = meta[meta["asin"].isin(item_map.keys())].copy()

    rev_path = proc_dir / "interactions.parquet"
    meta_path = proc_dir / "item_multimodal.parquet"
    umap_path = proc_dir / "user_id_map.json"
    imap_path = proc_dir / "item_id_map.json"

    rev.to_parquet(rev_path, index=False)
    meta_sub.to_parquet(meta_path, index=False)
    umap_path.write_text(json.dumps(user_map, ensure_ascii=False, indent=2), encoding="utf-8")
    imap_path.write_text(json.dumps(item_map, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "n_interactions": int(len(rev)),
        "n_users": int(rev["user_idx"].nunique()),
        "n_items": int(rev["item_idx"].nunique()),
        "n_items_with_meta": int(meta_sub["asin"].nunique()),
        "items_with_image_url": int((meta_sub["imUrl"].str.len() > 0).sum()),
        "items_with_title": int((meta_sub["title"].str.len() > 0).sum()),
        "split_counts": rev["split"].value_counts().to_dict(),
    }
    (proc_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "interactions": rev_path,
        "item_multimodal": meta_path,
        "user_id_map": umap_path,
        "item_id_map": imap_path,
        "stats": stats,
    }
