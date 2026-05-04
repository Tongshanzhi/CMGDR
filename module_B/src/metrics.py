from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def ranking_metrics_for_user(recommended: np.ndarray, targets: np.ndarray, topk: Iterable[int]) -> dict[str, float]:
    target_set = set(int(x) for x in targets.tolist())
    metrics: dict[str, float] = {}
    first_relevant_rank = None
    for rank, item_idx in enumerate(recommended.tolist(), start=1):
        if int(item_idx) in target_set:
            first_relevant_rank = rank
            break

    for k in topk:
        top_items = recommended[:k]
        hits = [idx for idx, item in enumerate(top_items.tolist(), start=1) if int(item) in target_set]
        recall = len(hits) / max(len(target_set), 1)
        hr = 1.0 if hits else 0.0
        if hits:
            dcg = sum(1.0 / math.log2(rank + 1.0) for rank in hits)
            ideal_hits = min(len(target_set), k)
            idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
            ndcg = dcg / max(idcg, 1e-8)
        else:
            ndcg = 0.0
        metrics[f"Recall@{k}"] = float(recall)
        metrics[f"HR@{k}"] = float(hr)
        metrics[f"NDCG@{k}"] = float(ndcg)
    metrics["MRR"] = 0.0 if first_relevant_rank is None else 1.0 / float(first_relevant_rank)
    return metrics


def aggregate_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {}
    keys = sorted(metric_dicts[0].keys())
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}


def cluster_exposure_gap(
    recommendations_by_user: dict[int, np.ndarray],
    visual_clusters: np.ndarray,
    topk: int,
) -> tuple[float, pd.DataFrame]:
    num_clusters = int(visual_clusters.max()) + 1
    exposure_counts = np.zeros(num_clusters, dtype=np.float64)
    total_slots = 0
    for items in recommendations_by_user.values():
        top_items = np.asarray(items[:topk], dtype=np.int64)
        exposure_counts += np.bincount(visual_clusters[top_items], minlength=num_clusters)
        total_slots += len(top_items)
    exposure_share = exposure_counts / max(total_slots, 1)
    catalog_share = np.bincount(visual_clusters, minlength=num_clusters) / len(visual_clusters)
    gap = np.abs(exposure_share - catalog_share)
    frame = pd.DataFrame(
        {
            "cluster_id": np.arange(num_clusters, dtype=np.int64),
            "catalog_share": catalog_share,
            "exposure_share": exposure_share,
            "exposure_gap": gap,
        }
    )
    return float(gap.mean()), frame


def cluster_calibration_gap(
    recommendations_by_user: dict[int, np.ndarray],
    targets_by_user: dict[int, np.ndarray],
    visual_clusters: np.ndarray,
    topk: int,
) -> tuple[float, pd.DataFrame]:
    num_clusters = int(visual_clusters.max()) + 1
    exposure_counts = np.zeros(num_clusters, dtype=np.float64)
    relevance_counts = np.zeros(num_clusters, dtype=np.float64)
    total_slots = 0
    total_targets = 0
    for user_idx, recommendations in recommendations_by_user.items():
        top_items = np.asarray(recommendations[:topk], dtype=np.int64)
        targets = np.asarray(targets_by_user[user_idx], dtype=np.int64)
        exposure_counts += np.bincount(visual_clusters[top_items], minlength=num_clusters)
        relevance_counts += np.bincount(visual_clusters[targets], minlength=num_clusters)
        total_slots += len(top_items)
        total_targets += len(targets)
    exposure_share = exposure_counts / max(total_slots, 1)
    target_share = relevance_counts / max(total_targets, 1)
    gap = np.abs(exposure_share - target_share)
    frame = pd.DataFrame(
        {
            "cluster_id": np.arange(num_clusters, dtype=np.int64),
            "target_share": target_share,
            "exposure_share": exposure_share,
            "calibration_gap": gap,
        }
    )
    return float(gap.mean()), frame


def cluster_probe_accuracy(
    embeddings: np.ndarray,
    visual_clusters: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = 42,
    max_iter: int = 300,
) -> float:
    if len(np.unique(visual_clusters)) < 2:
        return 0.0
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        visual_clusters,
        test_size=test_ratio,
        random_state=random_state,
        stratify=visual_clusters,
    )
    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))
