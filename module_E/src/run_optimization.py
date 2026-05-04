#!/usr/bin/env python3
"""
Phase 2: Targeted optimization based on LOO ablation insights.

Key findings from Phase 1:
- Item-item graph is the dominant contributor (+20%)
- Text alone provides no benefit (needs alignment mechanism)
- Contrastive loss HURTS when combined with full debiasing (1.20x -> 1.13x)
- Best so far: loo_text_ii = 0.1680 (1.20x), no debiasing

Strategy: Disentangle contrastive from debiasing, optimize each separately.
Target: MGCN-level 1.45x = R@10 ~0.203
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MODULE_B = str(Path(__file__).resolve().parents[2] / "module_B")
if _MODULE_B not in sys.path:
    sys.path.insert(0, _MODULE_B)

from run_experiments import run_single_experiment
from src.utils import PACKAGE_ROOT, load_config, save_json, ensure_dir


def main():
    config = load_config(PACKAGE_ROOT / "config" / "model.yaml")
    seed = 42
    results = []
    result_dir = ensure_dir(PACKAGE_ROOT / "results")

    experiments = [
        # === Group A: No debiasing, pure multimodal optimization ===

        # A1: text+ii, contrastive only (no debiasing), low temp
        {
            "name": "opt_A1_contrastive_only_temp01",
            "mode": "lightgcn",  # no visual debiasing
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 50,
                "learning_rate": 0.001,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0.1,
                    "contrastive_temperature": 0.1, "text_consistency": 0,
                },
            },
        },
        # A2: text+ii, contrastive with higher weight
        {
            "name": "opt_A2_contrastive_w02",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 50,
                "learning_rate": 0.001,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0.2,
                    "contrastive_temperature": 0.07, "text_consistency": 0,
                },
            },
        },
        # A3: text+ii, contrastive + text_consistency, no debiasing
        {
            "name": "opt_A3_contrastive_textcon",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 50,
                "learning_rate": 0.001,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0.1,
                    "contrastive_temperature": 0.1, "text_consistency": 0.05,
                },
            },
        },
        # A4: emb=256, text+ii, contrastive, no debiasing
        {
            "name": "opt_A4_emb256_contrastive",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 50,
                "learning_rate": 0.001,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0.1,
                    "contrastive_temperature": 0.1, "text_consistency": 0,
                },
            },
        },
        # A5: emb=256, text+ii, more epochs, lower lr
        {
            "name": "opt_A5_emb256_60ep_lr0005",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 60,
                "learning_rate": 0.0005,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0.15,
                    "contrastive_temperature": 0.07, "text_consistency": 0.03,
                },
            },
        },

        # === Group B: Very light debiasing + multimodal ===

        # B1: residual-only debiasing (no adversarial, no counterfactual) + multimodal
        {
            "name": "opt_B1_residual_only_mm",
            "mode": "residual",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 50,
                "learning_rate": 0.001,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0.1,
                    "contrastive_temperature": 0.1, "text_consistency": 0,
                },
            },
        },
        # B2: minimal debiasing (tiny adv=0.001) + multimodal
        {
            "name": "opt_B2_minimal_debias_mm",
            "mode": "full",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 50,
                "learning_rate": 0.001,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.3, "adversarial": 0.001, "counterfactual": 0.1,
                    "orthogonality": 0.02, "contrastive": 0.1,
                    "contrastive_temperature": 0.1, "text_consistency": 0,
                },
            },
        },
        # B3: residual-only + emb=256 + multimodal, no contrastive
        {
            "name": "opt_B3_residual_256_nocontr",
            "mode": "residual",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 50,
                "learning_rate": 0.001,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },

        # === Group C: Capacity & training schedule ===

        # C1: emb=256, text+ii, longer training (80ep), lr schedule via lower lr
        {
            "name": "opt_C1_emb256_80ep",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 80,
                "learning_rate": 0.001,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # C2: emb=128 higher lr, text+ii, 60ep
        {
            "name": "opt_C2_lr002_60ep",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 60,
                "learning_rate": 0.002,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
    ]

    for exp in experiments:
        cfg = copy.deepcopy(config)
        for k, v in exp["overrides"].items():
            cfg[k] = v

        t0 = time.time()
        try:
            result = run_single_experiment(
                config=cfg,
                mode=exp["mode"],
                seed=seed,
                suffix=exp["name"],
                use_loo=True,
                num_epochs=cfg.get("num_epochs", 50),
            )
            elapsed = time.time() - t0
            result["experiment"] = exp["name"]
            result["elapsed_sec"] = round(elapsed, 1)
            result["config_snapshot"] = {
                "mode": exp["mode"],
                "embedding_dim": cfg["embedding_dim"],
                "learning_rate": cfg["learning_rate"],
                "num_epochs": cfg["num_epochs"],
                "loss_weights": cfg["loss_weights"],
            }
            results.append(result)

            print(f"\n>>> {exp['name']}: R@10={result.get('Recall@10', 0):.4f}, "
                  f"R@20={result.get('Recall@20', 0):.4f}, "
                  f"N@10={result.get('NDCG@10', 0):.4f}, "
                  f"N@20={result.get('NDCG@20', 0):.4f}, "
                  f"time={elapsed:.0f}s\n")
        except Exception as e:
            print(f"\n>>> {exp['name']} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            results.append({"experiment": exp["name"], "error": str(e)})

        save_json(result_dir / "optimization_results.json", results)

    # Summary
    print("\n" + "=" * 100)
    print("OPTIMIZATION RESULTS (LOO, Sampled 1+999)")
    print("=" * 100)
    # Reference baseline from Phase 1
    baseline_r10 = 0.1402  # loo_lightgcn
    best_phase1 = 0.1680  # loo_text_ii
    print(f"Reference: LightGCN baseline R@10={baseline_r10:.4f}, Phase1 best={best_phase1:.4f} (1.20x)")
    print(f"Target: MGCN-level 1.45x = R@10~{baseline_r10*1.45:.4f}")
    print()
    print(f"{'Experiment':<35} {'R@10':>8} {'R@20':>8} {'N@10':>8} {'N@20':>8} {'vs base':>8}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['experiment']:<35} FAILED: {r['error']}")
        else:
            r10 = r.get('Recall@10', 0)
            ratio = f"{r10/baseline_r10:.2f}x"
            print(f"{r['experiment']:<35} {r10:>8.4f} {r.get('Recall@20',0):>8.4f} "
                  f"{r.get('NDCG@10',0):>8.4f} {r.get('NDCG@20',0):>8.4f} {ratio:>8}")
    print("=" * 100)

    save_json(result_dir / "optimization_results.json", results)
    print(f"\nResults saved to: {result_dir / 'optimization_results.json'}")


if __name__ == "__main__":
    main()
