"""Section 7.9 — Stratified vs uniform negative sampling ablation.

Train CMGDR-Full at seed 42 with `stratified_sampling=False` (uniform
negative sampler) holding all other hyperparameters fixed. Compare to the
multi-seed CMGDR-Full result that already uses stratified=True.
"""
from __future__ import annotations

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


def main():
    seed = 42
    overrides = {
        "use_item_item_graph": True,
        "embedding_dim": 256,
        "learning_rate": 0.002,
        "stratified_sampling": False,  # the ablation knob
        "loss_weights": {
            "residual": 0.5, "adversarial": 0.01, "counterfactual": 0.5,
            "orthogonality": 0.05, "contrastive": 0,
            "contrastive_temperature": 0.2, "text_consistency": 0,
        },
    }
    t0 = time.time()
    out = run_cmgdr_experiment(
        config=CFG, mode="full", seed=seed,
        suffix="CMGDR-Full_uniform_neg", num_epochs=40, overrides=overrides,
    )
    out.update({
        "experiment": "uniform_neg_ablation",
        "stratified_sampling": False,
        "time_sec": time.time() - t0,
        "seed": seed,
    })
    save_json(NEW / "stratified_ablation_raw.json", [out])
    print(f"Saved: {NEW / 'stratified_ablation_raw.json'}")


if __name__ == "__main__":
    main()
