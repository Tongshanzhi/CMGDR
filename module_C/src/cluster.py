"""K-means visual prototype clustering on extracted embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cluster(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root)
    cfg = _load_config(Path(config_path) if config_path else root / "config" / "visual.yaml")
    cl = cfg["cluster"]
    ext = cfg["extract"]

    # Load embeddings
    emb_path = root / ext["output_path"]
    embeddings = np.load(emb_path).astype(np.float32)
    n_items, visual_dim = embeddings.shape
    print(f"Loaded embeddings: {embeddings.shape} from {emb_path}")

    # Clustering
    K = cl["num_clusters"]
    method = cl.get("method", "kmeans")
    seed = cl.get("random_state", 42)
    n_init = cl.get("n_init", 10)

    print(f"Running {method} with K={K}, n_init={n_init}, seed={seed}")
    if method == "minibatch_kmeans":
        estimator = MiniBatchKMeans(
            n_clusters=K, n_init=n_init, random_state=seed, batch_size=1024
        )
    else:
        estimator = KMeans(n_clusters=K, n_init=n_init, random_state=seed)

    clusters = estimator.fit_predict(embeddings).astype(np.int64)

    # Compute prototypes (cluster centroids from actual data, not estimator.cluster_centers_)
    prototypes = np.zeros((K, visual_dim), dtype=np.float32)
    for k in range(K):
        mask = clusters == k
        if np.any(mask):
            prototypes[k] = embeddings[mask].mean(axis=0)

    # Cluster sizes
    cluster_sizes = [int((clusters == k).sum()) for k in range(K)]
    size_std = float(np.std(cluster_sizes))

    # Silhouette score (sample for speed if large)
    sil = -1.0
    if n_items > 500:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n_items, size=min(10000, n_items), replace=False)
        sil = float(silhouette_score(embeddings[sample_idx], clusters[sample_idx]))
    else:
        sil = float(silhouette_score(embeddings, clusters))
    print(f"Silhouette score: {sil:.4f}")

    # Write clusters CSV
    clusters_path = root / cl["output_clusters"]
    clusters_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "item_idx": np.arange(n_items, dtype=np.int64),
        "cluster_id": clusters,
    }).to_csv(clusters_path, index=False)
    print(f"Clusters saved: {clusters_path}")

    # Write prototypes
    proto_path = root / cl["output_prototypes"]
    np.save(proto_path, prototypes)
    print(f"Prototypes saved: {proto_path} shape={prototypes.shape}")

    # Write summary
    summary = {
        "num_clusters": K,
        "method": method,
        "n_init": n_init,
        "random_state": seed,
        "n_items": n_items,
        "visual_dim": visual_dim,
        "cluster_sizes": cluster_sizes,
        "size_min": int(min(cluster_sizes)),
        "size_max": int(max(cluster_sizes)),
        "size_mean": float(np.mean(cluster_sizes)),
        "size_std": round(size_std, 2),
        "inertia": float(estimator.inertia_),
        "silhouette_score": round(sil, 4),
    }
    summary_path = root / cl["output_summary"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved: {summary_path}")

    return summary
