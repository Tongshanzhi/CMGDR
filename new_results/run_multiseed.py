"""Section 7.1 — Multi-seed robustness.

Re-run MM-LightGCN, FREEDOM, CMGDR-Residual, CMGDR-Full at seeds {123, 2024}
(seed 42 already in shared_data/evaluation/full_evaluation_results.json
 / shared_data/evaluation/baseline_results.json), then merge with seed 42 and
report mean ± std for the headline metrics.
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
from run_baselines import run_baseline as run_baseline_experiment
from src.models.baselines import FREEDOMModel


CFG = load_config(PACKAGE_ROOT / "config" / "model.yaml")
NEW = ensure_dir(ROOT / "new_results")


# ---- Experiment definitions ----
CMGDR_VARIANTS = [
    {
        "name": "MM-LightGCN",
        "mode": "lightgcn",
        "epochs": 40,
        "overrides": {
            "use_item_item_graph": True,
            "embedding_dim": 128,
            "learning_rate": 0.001,
            "loss_weights": {
                "residual": 0, "adversarial": 0, "counterfactual": 0,
                "orthogonality": 0, "contrastive": 0,
                "contrastive_temperature": 0.2, "text_consistency": 0,
            },
        },
    },
    {
        "name": "CMGDR-Residual",
        "mode": "residual",
        "epochs": 40,
        "overrides": {
            "use_item_item_graph": True,
            "embedding_dim": 256,
            "learning_rate": 0.002,
            "loss_weights": {
                "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                "orthogonality": 0.05, "contrastive": 0,
                "contrastive_temperature": 0.2, "text_consistency": 0,
            },
        },
    },
    {
        "name": "CMGDR-Full",
        "mode": "full",
        "epochs": 40,
        "overrides": {
            "use_item_item_graph": True,
            "embedding_dim": 256,
            "learning_rate": 0.002,
            "loss_weights": {
                "residual": 0.5, "adversarial": 0.01, "counterfactual": 0.5,
                "orthogonality": 0.05, "contrastive": 0,
                "contrastive_temperature": 0.2, "text_consistency": 0,
            },
        },
    },
]

# FREEDOM is in the baselines runner; we reuse it.
FREEDOM_KWARGS = {"k_neighbors": 10}


def run_one_cmgdr(spec, seed):
    t0 = time.time()
    out = run_cmgdr_experiment(
        config=CFG, mode=spec["mode"], seed=seed,
        suffix=f"{spec['name']}_seed{seed}", num_epochs=spec["epochs"],
        overrides=spec.get("overrides"),
    )
    out["time_sec"] = time.time() - t0
    out["seed"] = seed
    out["model"] = spec["name"]
    return out


def run_one_freedom(seed):
    t0 = time.time()
    out = run_baseline_experiment(
        FREEDOMModel, f"FREEDOM_seed{seed}", CFG, seed,
        num_epochs=40, extra_kwargs=FREEDOM_KWARGS,
    )
    out["time_sec"] = time.time() - t0
    out["seed"] = seed
    out["model"] = "FREEDOM"
    return out


def main():
    NEW_SEEDS = [42, 123, 2024]
    all_runs = []
    log_path = NEW / "multiseed_log.jsonl"
    log_path.unlink(missing_ok=True)

    print(f"\n========== MULTI-SEED ROBUSTNESS — seeds {NEW_SEEDS} ==========")
    for seed in NEW_SEEDS:
        print(f"\n###### Seed {seed} ######")
        for spec in CMGDR_VARIANTS:
            r = run_one_cmgdr(spec, seed)
            all_runs.append(r)
            with open(log_path, "a") as f:
                f.write(json.dumps(r, default=float) + "\n")

        r = run_one_freedom(seed)
        all_runs.append(r)
        with open(log_path, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")

    save_json(NEW / "multiseed_raw.json", all_runs)
    print(f"\nSaved raw results: {NEW / 'multiseed_raw.json'}")


if __name__ == "__main__":
    main()
