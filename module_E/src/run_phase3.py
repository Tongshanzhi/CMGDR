#!/usr/bin/env python3
"""
Phase 3: Push toward MGCN-level performance (1.45x).

Best from Phase 2: opt_B3 (residual + emb256 + text + ii-graph) = 0.1780 (1.27x)
Target: 0.2033 (1.45x)

Strategy:
1. Higher capacity (emb=256/512)
2. Residual-only debiasing (proven to help)
3. More GCN layers
4. Longer training with careful LR
5. Combined residual + contrastive with optimal weights
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

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
        # P3-1: Best from P2 (B3) but with more epochs (80)
        {
            "name": "p3_residual_256_80ep",
            "mode": "residual",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 80,
                "learning_rate": 0.001,
                "num_layers": 3,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P3-2: emb=256 + residual + contrastive (combine B1+B3 ideas)
        {
            "name": "p3_residual_256_contr",
            "mode": "residual",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 60,
                "learning_rate": 0.001,
                "num_layers": 3,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0.1,
                    "contrastive_temperature": 0.1, "text_consistency": 0,
                },
            },
        },
        # P3-3: emb=512 + residual (even higher capacity)
        {
            "name": "p3_residual_512",
            "mode": "residual",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 512,
                "num_epochs": 50,
                "learning_rate": 0.0005,
                "num_layers": 3,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P3-4: emb=256, 4 GCN layers (deeper propagation)
        {
            "name": "p3_residual_256_4layers",
            "mode": "residual",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 60,
                "learning_rate": 0.001,
                "num_layers": 4,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P3-5: emb=256, residual + light contrastive + low lr + 80ep
        {
            "name": "p3_best_combo_80ep",
            "mode": "residual",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 80,
                "learning_rate": 0.0008,
                "num_layers": 3,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0.05,
                    "contrastive_temperature": 0.1, "text_consistency": 0,
                },
            },
        },
        # P3-6: emb=256, residual with stronger weight + no contrastive
        {
            "name": "p3_residual_strong_256",
            "mode": "residual",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 60,
                "learning_rate": 0.001,
                "num_layers": 3,
                "loss_weights": {
                    "residual": 1.0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.1, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P3-7: pure LightGCN + text + ii + emb=256 + 4 layers (capacity upper bound)
        {
            "name": "p3_lightgcn_256_4layers",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 60,
                "learning_rate": 0.001,
                "num_layers": 4,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P3-8: emb=256, residual + lr=0.002 + 40ep (faster convergence)
        {
            "name": "p3_residual_256_lr002",
            "mode": "residual",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 40,
                "learning_rate": 0.002,
                "num_layers": 3,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
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
                "num_layers": cfg["num_layers"],
                "loss_weights": cfg["loss_weights"],
            }
            results.append(result)

            baseline_r10 = 0.1402
            r10 = result.get('Recall@10', 0)
            print(f"\n>>> {exp['name']}: R@10={r10:.4f} ({r10/baseline_r10:.2f}x), "
                  f"R@20={result.get('Recall@20', 0):.4f}, "
                  f"N@10={result.get('NDCG@10', 0):.4f}, "
                  f"time={elapsed:.0f}s\n")
        except Exception as e:
            print(f"\n>>> {exp['name']} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            results.append({"experiment": exp["name"], "error": str(e)})

        save_json(result_dir / "phase3_results.json", results)

    # Summary
    baseline_r10 = 0.1402
    print("\n" + "=" * 100)
    print("PHASE 3 RESULTS (LOO, Sampled 1+999)")
    print("=" * 100)
    print(f"Target: MGCN 1.45x = {baseline_r10*1.45:.4f}")
    print()
    print(f"{'Experiment':<35} {'R@10':>8} {'R@20':>8} {'N@10':>8} {'N@20':>8} {'ratio':>8}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x.get('Recall@10', 0), reverse=True):
        if "error" not in r:
            r10 = r.get('Recall@10', 0)
            print(f"  {r['experiment']:<35} {r10:>7.4f} {r.get('Recall@20',0):>8.4f} "
                  f"{r.get('NDCG@10',0):>8.4f} {r.get('NDCG@20',0):>8.4f} {r10/baseline_r10:>7.2f}x")
    print("=" * 100)

    save_json(result_dir / "phase3_results.json", results)


if __name__ == "__main__":
    main()
