from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .utils import ensure_dir, resolve_path, save_json


@dataclass
class VisualArtifacts:
    features: np.ndarray
    clusters: np.ndarray
    prototypes: np.ndarray
    feature_path: Path
    cluster_path: Path
    summary: dict[str, Any]


def load_visual_features(path: str | Path, expected_items: int) -> np.ndarray:
    feature_path = Path(path)
    features = np.load(feature_path)
    if features.ndim != 2:
        raise ValueError(f"Visual embeddings must be rank-2, got shape={features.shape}")
    if features.shape[0] != expected_items:
        raise ValueError(
            f"Visual embedding rows {features.shape[0]} do not match expected n_items={expected_items}"
        )
    return features.astype(np.float32)


def load_visual_clusters(path: str | Path, expected_items: int) -> np.ndarray:
    cluster_path = Path(path)
    if cluster_path.suffix.lower() == ".csv":
        frame = pd.read_csv(cluster_path)
        if {"item_idx", "cluster_id"}.issubset(frame.columns):
            frame = frame.sort_values("item_idx")
            clusters = frame["cluster_id"].to_numpy(dtype=np.int64)
        elif len(frame.columns) == 1:
            clusters = frame.iloc[:, 0].to_numpy(dtype=np.int64)
        else:
            raise ValueError(f"Unsupported cluster CSV columns: {list(frame.columns)}")
    elif cluster_path.suffix.lower() == ".json":
        payload = json.loads(cluster_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            clusters = np.asarray(payload, dtype=np.int64)
        elif isinstance(payload, dict):
            ordered = sorted((int(k), int(v)) for k, v in payload.items())
            clusters = np.asarray([v for _, v in ordered], dtype=np.int64)
        else:
            raise ValueError("Cluster JSON must be a list or {item_idx: cluster_id} mapping")
    else:
        raise ValueError(f"Unsupported cluster file format: {cluster_path}")
    if len(clusters) != expected_items:
        raise ValueError(f"Cluster assignments {len(clusters)} do not match expected n_items={expected_items}")
    return clusters


def build_kmeans_clusters(features: np.ndarray, num_clusters: int, random_state: int = 42) -> np.ndarray:
    estimator = KMeans(n_clusters=num_clusters, n_init=10, random_state=random_state)
    return estimator.fit_predict(features).astype(np.int64)


def compute_cluster_prototypes(features: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    num_clusters = int(clusters.max()) + 1
    prototypes = np.zeros((num_clusters, features.shape[1]), dtype=np.float32)
    for cluster_id in range(num_clusters):
        member_mask = clusters == cluster_id
        if not np.any(member_mask):
            continue
        prototypes[cluster_id] = features[member_mask].mean(axis=0)
    return prototypes


def persist_clusters(path: str | Path, clusters: np.ndarray) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    frame = pd.DataFrame({"item_idx": np.arange(len(clusters), dtype=np.int64), "cluster_id": clusters})
    frame.to_csv(target, index=False)
    return target


def prepare_visual_artifacts(
    config: dict[str, Any],
    package_root: str | Path,
    expected_items: int,
    seed: int,
) -> VisualArtifacts:
    root = Path(package_root)
    feature_path = resolve_path(root, config.get("visual_feature_path"))
    if feature_path is None or not feature_path.exists():
        raise FileNotFoundError(
            "Visual embeddings are required for visual-first CMGDR. "
            f"Expected file: {feature_path}"
        )

    features = load_visual_features(feature_path, expected_items)
    requested_cluster_path = resolve_path(root, config.get("visual_cluster_path"))
    if requested_cluster_path is not None and requested_cluster_path.exists():
        clusters = load_visual_clusters(requested_cluster_path, expected_items)
        cluster_path = requested_cluster_path
        source = "provided"
    else:
        artifact_dir = ensure_dir(resolve_path(root, config.get("artifact_dir", "artifacts")))
        cluster_path = artifact_dir / (
            f"{config.get('category', 'dataset')}_visual_clusters_k{int(config.get('num_visual_clusters', 32))}.csv"
        )
        clusters = build_kmeans_clusters(features, int(config.get("num_visual_clusters", 32)), random_state=seed)
        persist_clusters(cluster_path, clusters)
        source = "kmeans_fallback"

    prototypes = compute_cluster_prototypes(features, clusters)
    summary = {
        "feature_path": str(feature_path),
        "cluster_path": str(cluster_path),
        "cluster_source": source,
        "n_items": int(expected_items),
        "feature_dim": int(features.shape[1]),
        "num_clusters": int(clusters.max()) + 1,
    }
    save_json(cluster_path.with_suffix(".summary.json"), summary)
    return VisualArtifacts(
        features=features,
        clusters=clusters,
        prototypes=prototypes,
        feature_path=feature_path,
        cluster_path=cluster_path,
        summary=summary,
    )
