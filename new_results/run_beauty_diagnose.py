"""Diagnostic: did CMGDR underperform on Beauty because of bad hyperparams?

We hold the dataset/code/protocol fixed and vary just one knob each:
  A. emb_dim 256 -> 128 (match MM-LightGCN capacity)
  B. lighter loss weights (residual=0.1, adv=0.001, cf=0.1)
  C. extra seed (123) at the headline config (variance check)
  D. Stronger residual weight (residual=1.0, adv=0.05, cf=0.5)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module_B"))

from src.utils import load_config, ensure_dir, save_json
from run_full_evaluation import run_experiment as run_cmgdr_experiment

BEAUTY_CFG = Path("/root/autodl-tmp/CMGDR_beauty/model_beauty.yaml")
NEW = ensure_dir(ROOT / "new_results")
LOG = NEW / "beauty_diagnose_log.jsonl"
LOG.unlink(missing_ok=True)


SPECS = [
    {
        "name": "CMGDR-Full_emb128",
        "seed": 42,
        "overrides": {
            "use_item_item_graph": True,
            "embedding_dim": 128,            # capacity-matched to MM-LightGCN
            "learning_rate": 0.002,
            "loss_weights": {
                "residual": 0.5, "adversarial": 0.01, "counterfactual": 0.5,
                "orthogonality": 0.05, "contrastive": 0,
                "contrastive_temperature": 0.2, "text_consistency": 0,
            },
        },
    },
    {
        "name": "CMGDR-Full_lighter_loss",
        "seed": 42,
        "overrides": {
            "use_item_item_graph": True,
            "embedding_dim": 256,
            "learning_rate": 0.002,
            "loss_weights": {
                "residual": 0.1, "adversarial": 0.001, "counterfactual": 0.1,
                "orthogonality": 0.01, "contrastive": 0,
                "contrastive_temperature": 0.2, "text_consistency": 0,
            },
        },
    },
    {
        "name": "CMGDR-Full_seed123",
        "seed": 123,                          # variance check
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
    {
        "name": "MM-LightGCN_seed123",
        "seed": 123,                          # MM-LightGCN variance counterpart
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
        "mode": "lightgcn",
    },
]


def main():
    cfg = load_config(BEAUTY_CFG)
    results = []
    for spec in SPECS:
        t0 = time.time()
        mode = spec.get("mode", "full")
        r = run_cmgdr_experiment(
            config=cfg, mode=mode, seed=spec["seed"],
            suffix=f"{spec['name']}_beauty", num_epochs=40,
            overrides=spec["overrides"],
        )
        r["time_sec"] = time.time() - t0
        r["seed"] = spec["seed"]
        r["model"] = spec["name"]
        r["dataset"] = "Amazon_Beauty"
        results.append(r)
        with open(LOG, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
        probe = r.get("probe_causal", r.get("probe_item_emb", 0))
        print(f">>> {spec['name']}: R@10={r.get('Recall@10', 0):.4f} "
              f"valid={r.get('best_valid_R@10', 0):.4f} "
              f"N@10={r.get('NDCG@10', 0):.4f} Probe={probe:.3f}")
    save_json(NEW / "beauty_diagnose_raw.json", results)
    print(f"Saved: {NEW / 'beauty_diagnose_raw.json'}")


if __name__ == "__main__":
    main()
