from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # type: ignore[override]
        """Fallback so the module remains importable without torch."""

        pass


REQUIRED_INTERACTION_COLUMNS = {
    "reviewerID",
    "asin",
    "user_idx",
    "item_idx",
    "overall",
    "unixReviewTime",
    "split",
}
REQUIRED_ITEM_COLUMNS = {"asin"}
REQUIRED_EDGE_COLUMNS = {"src", "dst", "rating", "time", "split"}


@dataclass
class UpstreamPaths:
    root: Path
    interactions: Path
    items: Path
    item_id_map: Path
    edges_train: Path
    graph_meta: Path


@dataclass
class EvalSplit:
    name: str
    user_ids: np.ndarray
    targets_by_user: dict[int, np.ndarray]
    mask_by_user: dict[int, np.ndarray]


@dataclass
class DataBundle:
    data_root: Path
    interactions: pd.DataFrame
    items: pd.DataFrame
    edges_train: pd.DataFrame
    graph_meta: dict[str, Any]
    item_id_map: dict[str, int]
    user_pos_all: dict[int, set[int]]
    user_pos_all_arrays: dict[int, np.ndarray]
    eval_splits: dict[str, EvalSplit]

    @property
    def n_users(self) -> int:
        return int(self.graph_meta["n_users"])

    @property
    def n_items(self) -> int:
        return int(self.graph_meta["n_items"])

    @property
    def n_nodes(self) -> int:
        return int(self.graph_meta["n_nodes"])

    @property
    def train_interactions(self) -> pd.DataFrame:
        return self.interactions[self.interactions["split"] == "train"].reset_index(drop=True)


class PairwiseTrainDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        n_items: int,
        user_pos_all: dict[int, set[int]],
        negatives_per_pos: int = 1,
        seed: int = 42,
        visual_clusters: np.ndarray | None = None,
        stratified_sampling: bool = False,
    ) -> None:
        train_df = interactions[interactions["split"] == "train"][["user_idx", "item_idx"]].copy()
        self.train_pairs = train_df.to_numpy(dtype=np.int64)
        self.n_items = int(n_items)
        self.user_pos_all = user_pos_all
        self.negatives_per_pos = max(int(negatives_per_pos), 1)
        self.rng = np.random.default_rng(seed)
        self.stratified = stratified_sampling and visual_clusters is not None

        if self.stratified and visual_clusters is not None:
            # Pre-group items by cluster for stratified sampling
            n_clusters = int(visual_clusters.max()) + 1
            self.cluster_items: list[np.ndarray] = []
            for c in range(n_clusters):
                self.cluster_items.append(np.where(visual_clusters == c)[0].astype(np.int64))
            # Prior P(V=v) = |cluster_v| / n_items
            self.cluster_prior = np.array([len(c) for c in self.cluster_items], dtype=np.float64)
            self.cluster_prior /= self.cluster_prior.sum()
            self.visual_clusters = visual_clusters
        else:
            self.visual_clusters = None

    def __len__(self) -> int:
        return int(len(self.train_pairs) * self.negatives_per_pos)

    def _sample_negative(self, user_idx: int) -> int:
        positives = self.user_pos_all[user_idx]
        if self.stratified and self.visual_clusters is not None:
            # Backdoor adjustment: sample cluster from P(V=v), then item from cluster
            for _ in range(50):
                c = int(self.rng.choice(len(self.cluster_items), p=self.cluster_prior))
                items_in_c = self.cluster_items[c]
                item_idx = int(items_in_c[self.rng.integers(0, len(items_in_c))])
                if item_idx not in positives:
                    return item_idx
        # Fallback: uniform
        while True:
            item_idx = int(self.rng.integers(0, self.n_items))
            if item_idx not in positives:
                return item_idx

    def __getitem__(self, index: int) -> dict[str, int]:
        pair = self.train_pairs[index // self.negatives_per_pos]
        user_idx = int(pair[0])
        pos_item_idx = int(pair[1])
        neg_item_idx = self._sample_negative(user_idx)
        return {
            "user_idx": user_idx,
            "pos_item_idx": pos_item_idx,
            "neg_item_idx": neg_item_idx,
        }


def resolve_upstream_paths(data_root: str | Path) -> UpstreamPaths:
    root = Path(data_root)
    return UpstreamPaths(
        root=root,
        interactions=root / "processed" / "interactions.parquet",
        items=root / "processed" / "item_multimodal.parquet",
        item_id_map=root / "processed" / "item_id_map.json",
        edges_train=root / "graph" / "edges_train.parquet",
        graph_meta=root / "graph" / "graph_meta.json",
    )


def validate_upstream_contract(data_root: str | Path) -> dict[str, Any]:
    paths = resolve_upstream_paths(data_root)
    required_paths = [
        paths.interactions,
        paths.items,
        paths.item_id_map,
        paths.edges_train,
        paths.graph_meta,
    ]
    missing_files = [str(path) for path in required_paths if not path.exists()]
    summary: dict[str, Any] = {
        "data_root": str(paths.root),
        "missing_files": missing_files,
        "ok": not missing_files,
    }
    if missing_files:
        return summary

    interactions = pd.read_parquet(paths.interactions)
    items = pd.read_parquet(paths.items)
    edges_train = pd.read_parquet(paths.edges_train)
    graph_meta = json.loads(paths.graph_meta.read_text(encoding="utf-8"))
    item_id_map = json.loads(paths.item_id_map.read_text(encoding="utf-8"))

    summary.update(
        {
            "interaction_columns_ok": REQUIRED_INTERACTION_COLUMNS.issubset(interactions.columns),
            "item_columns_ok": REQUIRED_ITEM_COLUMNS.issubset(items.columns),
            "edge_columns_ok": REQUIRED_EDGE_COLUMNS.issubset(edges_train.columns),
            "n_users": int(graph_meta.get("n_users", -1)),
            "n_items": int(graph_meta.get("n_items", -1)),
            "n_edges_train": int(len(edges_train)),
            "n_item_map": int(len(item_id_map)),
        }
    )
    summary["ok"] = bool(
        summary["interaction_columns_ok"]
        and summary["item_columns_ok"]
        and summary["edge_columns_ok"]
        and not missing_files
    )
    return summary


def _read_item_frame(paths: UpstreamPaths, n_items: int) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_items = pd.read_parquet(paths.items)
    item_id_map = json.loads(paths.item_id_map.read_text(encoding="utf-8"))
    item_lookup = pd.DataFrame(
        {
            "asin": list(item_id_map.keys()),
            "item_idx": list(item_id_map.values()),
        }
    )
    item_lookup["item_idx"] = item_lookup["item_idx"].astype(np.int64)
    items = item_lookup.merge(raw_items, on="asin", how="left", suffixes=("", "_raw"))
    items = items.sort_values("item_idx").reset_index(drop=True)
    if len(items) != n_items:
        raise ValueError(f"Item map count {len(items)} does not match graph_meta n_items={n_items}")
    for column in ["title", "description", "imUrl", "brand"]:
        if column in items.columns:
            items[column] = items[column].fillna("")
    return items, item_id_map


def _build_user_positive_maps(interactions: pd.DataFrame) -> tuple[dict[int, set[int]], dict[int, np.ndarray]]:
    grouped = interactions.groupby("user_idx")["item_idx"].agg(list)
    user_pos_all: dict[int, set[int]] = {}
    user_pos_arrays: dict[int, np.ndarray] = {}
    for user_idx, item_list in grouped.items():
        sorted_items = np.asarray(sorted(set(int(x) for x in item_list)), dtype=np.int64)
        user_pos_all[int(user_idx)] = set(int(x) for x in sorted_items.tolist())
        user_pos_arrays[int(user_idx)] = sorted_items
    return user_pos_all, user_pos_arrays


def _build_eval_split(interactions: pd.DataFrame, split_name: str) -> EvalSplit:
    target_df = interactions[interactions["split"] == split_name]
    target_groups = target_df.groupby("user_idx")["item_idx"].agg(list)
    targets_by_user: dict[int, np.ndarray] = {}
    mask_by_user: dict[int, np.ndarray] = {}
    all_groups = interactions.groupby("user_idx")["item_idx"].agg(list)
    for user_idx, target_items in target_groups.items():
        user = int(user_idx)
        target_arr = np.asarray(sorted(set(int(x) for x in target_items)), dtype=np.int64)
        all_arr = np.asarray(sorted(set(int(x) for x in all_groups[user_idx])), dtype=np.int64)
        other_seen = np.setdiff1d(all_arr, target_arr, assume_unique=False)
        targets_by_user[user] = target_arr
        mask_by_user[user] = other_seen
    user_ids = np.asarray(sorted(targets_by_user.keys()), dtype=np.int64)
    return EvalSplit(
        name=split_name,
        user_ids=user_ids,
        targets_by_user=targets_by_user,
        mask_by_user=mask_by_user,
    )


def build_symmetric_edge_index(edges_train: pd.DataFrame) -> np.ndarray:
    src = edges_train["src"].to_numpy(dtype=np.int64)
    dst = edges_train["dst"].to_numpy(dtype=np.int64)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    return np.stack([row, col], axis=0)


def load_data_bundle(data_root: str | Path) -> DataBundle:
    contract = validate_upstream_contract(data_root)
    if not contract["ok"]:
        raise FileNotFoundError(f"Upstream data contract invalid: {contract}")

    paths = resolve_upstream_paths(data_root)
    interactions = pd.read_parquet(paths.interactions)
    interactions["user_idx"] = interactions["user_idx"].astype(np.int64)
    interactions["item_idx"] = interactions["item_idx"].astype(np.int64)
    interactions["unixReviewTime"] = interactions["unixReviewTime"].astype(np.int64)
    interactions = interactions.sort_values(["unixReviewTime", "user_idx", "item_idx"]).reset_index(drop=True)
    graph_meta = json.loads(paths.graph_meta.read_text(encoding="utf-8"))
    items, item_id_map = _read_item_frame(paths, int(graph_meta["n_items"]))
    edges_train = pd.read_parquet(paths.edges_train)
    edges_train["src"] = edges_train["src"].astype(np.int64)
    edges_train["dst"] = edges_train["dst"].astype(np.int64)

    user_pos_all, user_pos_all_arrays = _build_user_positive_maps(interactions)
    eval_splits = {
        "valid": _build_eval_split(interactions, "valid"),
        "test": _build_eval_split(interactions, "test"),
    }

    return DataBundle(
        data_root=paths.root,
        interactions=interactions,
        items=items,
        edges_train=edges_train,
        graph_meta=graph_meta,
        item_id_map=item_id_map,
        user_pos_all=user_pos_all,
        user_pos_all_arrays=user_pos_all_arrays,
        eval_splits=eval_splits,
    )
