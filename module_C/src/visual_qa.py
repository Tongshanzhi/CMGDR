"""Quality assurance: validate artifacts, generate visualizations and report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.manifold import TSNE


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_embeddings(emb_path: Path, n_items: int) -> dict[str, Any]:
    checks: dict[str, Any] = {"path": str(emb_path), "exists": emb_path.exists()}
    if not emb_path.exists():
        checks["error"] = "file not found"
        return checks

    emb = np.load(emb_path)
    checks["shape"] = list(emb.shape)
    checks["dtype"] = str(emb.dtype)
    checks["ndim_ok"] = emb.ndim == 2
    checks["n_rows_ok"] = emb.shape[0] == n_items
    checks["has_nan"] = bool(np.any(np.isnan(emb)))
    checks["has_inf"] = bool(np.any(np.isinf(emb)))
    zero_rows = int(np.all(emb == 0, axis=1).sum())
    checks["zero_rows"] = zero_rows
    checks["zero_row_pct"] = round(100.0 * zero_rows / max(n_items, 1), 2)

    # Norm statistics for non-zero rows
    nonzero_mask = ~np.all(emb == 0, axis=1)
    if np.any(nonzero_mask):
        norms = np.linalg.norm(emb[nonzero_mask], axis=1)
        checks["norm_mean"] = round(float(norms.mean()), 4)
        checks["norm_std"] = round(float(norms.std()), 4)
        checks["norm_min"] = round(float(norms.min()), 4)
        checks["norm_max"] = round(float(norms.max()), 4)

    checks["ok"] = (
        checks["ndim_ok"]
        and checks["n_rows_ok"]
        and not checks["has_nan"]
        and not checks["has_inf"]
    )
    return checks


def _check_clusters(cluster_path: Path, n_items: int) -> dict[str, Any]:
    checks: dict[str, Any] = {"path": str(cluster_path), "exists": cluster_path.exists()}
    if not cluster_path.exists():
        checks["error"] = "file not found"
        return checks

    df = pd.read_csv(cluster_path)
    checks["columns"] = list(df.columns)
    checks["has_required_cols"] = {"item_idx", "cluster_id"}.issubset(df.columns)
    checks["n_rows"] = len(df)
    checks["n_rows_ok"] = len(df) == n_items

    if checks["has_required_cols"]:
        idx_set = set(df["item_idx"].tolist())
        expected_set = set(range(n_items))
        checks["item_idx_complete"] = idx_set == expected_set
        checks["item_idx_duplicates"] = int(df["item_idx"].duplicated().sum())
        checks["cluster_id_min"] = int(df["cluster_id"].min())
        checks["cluster_id_max"] = int(df["cluster_id"].max())
        checks["num_clusters"] = int(df["cluster_id"].nunique())
        checks["has_nan"] = bool(df[["item_idx", "cluster_id"]].isna().any().any())
    else:
        checks["item_idx_complete"] = False

    checks["ok"] = (
        checks["has_required_cols"]
        and checks["n_rows_ok"]
        and checks.get("item_idx_complete", False)
        and not checks.get("has_nan", True)
        and checks.get("item_idx_duplicates", 1) == 0
    )
    return checks


def _plot_cluster_distribution(cluster_path: Path, output_path: Path, K: int) -> None:
    df = pd.read_csv(cluster_path)
    counts = df["cluster_id"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.axhline(y=counts.values.mean(), color="red", linestyle="--", linewidth=1,
               label=f"mean={counts.values.mean():.0f}")
    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("Item Count", fontsize=12)
    ax.set_title(f"Visual Cluster Size Distribution (K={K})", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Cluster distribution plot saved: {output_path}")


def _plot_tsne(
    emb_path: Path, cluster_path: Path, output_path: Path,
    sample_size: int, perplexity: int,
) -> None:
    embeddings = np.load(emb_path)
    clusters_df = pd.read_csv(cluster_path).sort_values("item_idx")
    clusters = clusters_df["cluster_id"].values

    n = len(embeddings)
    rng = np.random.default_rng(42)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
    else:
        idx = np.arange(n)

    emb_sample = embeddings[idx]
    cls_sample = clusters[idx]

    # Skip zero-vector rows for t-SNE
    nonzero = ~np.all(emb_sample == 0, axis=1)
    emb_nz = emb_sample[nonzero]
    cls_nz = cls_sample[nonzero]

    print(f"Running t-SNE on {len(emb_nz)} samples (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(emb_nz)

    fig, ax = plt.subplots(figsize=(10, 10))
    K = int(cls_nz.max()) + 1
    cmap = plt.colormaps.get_cmap("tab20").resampled(K)
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=cls_nz, cmap=cmap, s=3, alpha=0.6, edgecolors="none"
    )
    ax.set_title(f"t-SNE of Visual Embeddings (colored by cluster, K={K})", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, label="Cluster ID")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"t-SNE plot saved: {output_path}")


def run_qa(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root)
    cfg = _load_config(Path(config_path) if config_path else root / "config" / "visual.yaml")
    up = cfg["upstream"]
    ext = cfg["extract"]
    cl = cfg["cluster"]
    qa = cfg["qa"]

    data_root = Path(up["data_root"])
    with open(data_root / up["item_id_map"], encoding="utf-8") as f:
        n_items = len(json.load(f))

    print(f"=== Visual QA: {n_items} items ===")

    # Check embeddings
    emb_path = root / ext["output_path"]
    emb_checks = _check_embeddings(emb_path, n_items)
    print(f"Embeddings: {'PASS' if emb_checks['ok'] else 'FAIL'} "
          f"(shape={emb_checks.get('shape')}, zero_rows={emb_checks.get('zero_rows')})")

    # Check clusters
    cluster_path = root / cl["output_clusters"]
    cluster_checks = _check_clusters(cluster_path, n_items)
    print(f"Clusters: {'PASS' if cluster_checks['ok'] else 'FAIL'} "
          f"(K={cluster_checks.get('num_clusters')}, rows={cluster_checks.get('n_rows')})")

    # Check prototypes
    proto_path = root / cl["output_prototypes"]
    proto_checks: dict[str, Any] = {"path": str(proto_path), "exists": proto_path.exists()}
    if proto_path.exists():
        proto = np.load(proto_path)
        proto_checks["shape"] = list(proto.shape)
        K = cluster_checks.get("num_clusters", -1)
        vis_dim = emb_checks.get("shape", [0, 0])[1] if emb_checks.get("shape") else 0
        proto_checks["shape_ok"] = list(proto.shape) == [K, vis_dim]
        proto_checks["ok"] = proto_checks["shape_ok"]
    else:
        proto_checks["ok"] = False
    print(f"Prototypes: {'PASS' if proto_checks['ok'] else 'FAIL'} (shape={proto_checks.get('shape')})")

    # Plot cluster distribution
    dist_plot_path = root / qa["output_cluster_dist_plot"]
    if cluster_checks["ok"]:
        _plot_cluster_distribution(cluster_path, dist_plot_path, cluster_checks["num_clusters"])

    # Plot t-SNE
    tsne_plot_path = root / qa["output_tsne_plot"]
    if emb_checks["ok"] and cluster_checks["ok"]:
        _plot_tsne(
            emb_path, cluster_path, tsne_plot_path,
            qa.get("tsne_sample_size", 5000),
            qa.get("tsne_perplexity", 30),
        )

    # Compile report
    overall_ok = emb_checks["ok"] and cluster_checks["ok"] and proto_checks["ok"]
    report = {
        "overall_ok": overall_ok,
        "n_items": n_items,
        "embeddings": emb_checks,
        "clusters": cluster_checks,
        "prototypes": proto_checks,
        "plots": {
            "cluster_distribution": str(dist_plot_path),
            "tsne": str(tsne_plot_path),
        },
    }
    report_path = root / qa["output_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nQA Report: {'ALL PASS' if overall_ok else 'ISSUES FOUND'}")
    print(f"Saved: {report_path}")
    return report
