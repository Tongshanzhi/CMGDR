"""Section 7.3 — Hyper-parameter sensitivity.

(a) K∈{16, 32, 64} for visual cluster count. K=32 is the headline run, so
    this script trains CMGDR-Full at K=16 and K=64.
(b) Loss-profile sweep: light / medium / heavy as defined in
    module_B/config/model.yaml's `sensitivity` block.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module_B"))

from src.utils import PACKAGE_ROOT, load_config, ensure_dir, save_json
from run_full_evaluation import run_experiment as run_cmgdr_experiment


CFG = load_config(PACKAGE_ROOT / "config" / "model.yaml")
NEW = ensure_dir(ROOT / "new_results")


def _override_for_K(K):
    """Recluster ViT embeddings at K visual clusters; the visual_cluster_path
    is pointed at a fresh per-K artifact so prepare_visual_artifacts re-runs
    KMeans (sklearn.KMeans(n_init=10, random_state=42)). Existing K=32 csv
    is preserved (different filename)."""
    return {
        "use_item_item_graph": True,
        "embedding_dim": 256,
        "learning_rate": 0.002,
        "num_visual_clusters": K,
        # Force prepare_visual_artifacts to re-cluster by pointing at a path
        # that does not exist; the helper writes the csv next to artifact_dir.
        "visual_cluster_path": f"new_results/clusters_k{K}.csv",
        "loss_weights": {
            "residual": 0.5, "adversarial": 0.01, "counterfactual": 0.5,
            "orthogonality": 0.05, "contrastive": 0,
            "contrastive_temperature": 0.2, "text_consistency": 0,
        },
    }


# Loss profiles taken from module_B/config/model.yaml — sensitivity block.
LOSS_PROFILES = {
    "light":  {"residual": 0.5, "adversarial": 0.1, "counterfactual": 0.25, "orthogonality": 0.05},
    "medium": {"residual": 1.0, "adversarial": 0.2, "counterfactual": 0.5,  "orthogonality": 0.1},
    "heavy":  {"residual": 2.0, "adversarial": 0.4, "counterfactual": 1.0,  "orthogonality": 0.2},
}


def _override_for_profile(name):
    p = LOSS_PROFILES[name]
    return {
        "use_item_item_graph": True,
        "embedding_dim": 256,
        "learning_rate": 0.002,
        "loss_weights": {**p, "contrastive": 0, "contrastive_temperature": 0.2,
                         "text_consistency": 0},
    }


def main():
    seed = 42
    results = []
    log = NEW / "sensitivity_log.jsonl"
    log.unlink(missing_ok=True)

    print("\n========== HYPERPARAM SENSITIVITY ==========")

    # Cluster count sweep
    for K in [16, 64]:  # K=32 is the headline run
        t0 = time.time()
        spec_name = f"CMGDR-Full_K{K}"
        out = run_cmgdr_experiment(
            config=CFG, mode="full", seed=seed,
            suffix=spec_name, num_epochs=40,
            overrides=_override_for_K(K),
        )
        out.update({
            "experiment": "cluster_K",
            "K": K,
            "time_sec": time.time() - t0,
            "seed": seed,
        })
        results.append(out)
        with open(log, "a") as f:
            f.write(json.dumps(out, default=float) + "\n")

    # Loss profile sweep
    for prof_name in ["light", "medium", "heavy"]:
        t0 = time.time()
        spec_name = f"CMGDR-Full_loss_{prof_name}"
        out = run_cmgdr_experiment(
            config=CFG, mode="full", seed=seed,
            suffix=spec_name, num_epochs=40,
            overrides=_override_for_profile(prof_name),
        )
        out.update({
            "experiment": "loss_profile",
            "profile": prof_name,
            "loss_weights": LOSS_PROFILES[prof_name],
            "time_sec": time.time() - t0,
            "seed": seed,
        })
        results.append(out)
        with open(log, "a") as f:
            f.write(json.dumps(out, default=float) + "\n")

    save_json(NEW / "sensitivity_raw.json", results)
    print(f"\nSaved: {NEW / 'sensitivity_raw.json'}")


if __name__ == "__main__":
    main()
