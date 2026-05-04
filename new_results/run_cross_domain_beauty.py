"""Section 7.2 — Cross-domain generalisation on Amazon Beauty.

Reuses run_full_evaluation.run_experiment but points at the Beauty workspace
(/root/autodl-tmp/CMGDR_beauty/shared_data). Trains MM-LightGCN, FREEDOM,
CMGDR-Residual, CMGDR-Full at seed 42 with the same hyper-parameters used in
the Sports run, and dumps numbers to new_results/cross_domain_beauty.json.

Real data: Amazon Reviews 2023 Beauty 5-core
(https://amazon-reviews-2023.github.io/), processed by Module A; ViT-B/16
features extracted by Module C; SBERT MiniLM text features by Module D;
co-purchase item-item graph from Module A. Identical evaluation protocol
(LOO sampled 1+999) and identical model code paths as the Sports headline run.
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
from run_baselines import run_baseline as run_baseline_experiment
from src.models.baselines import FREEDOMModel


# Beauty-specific config — overrides the default Sports paths
BEAUTY_CFG_PATH = Path("/root/autodl-tmp/CMGDR_beauty/model_beauty.yaml")
NEW = ensure_dir(ROOT / "new_results")
LOG = NEW / "cross_domain_beauty_log.jsonl"
LOG.unlink(missing_ok=True)


SPECS = [
    {
        "name": "MM-LightGCN",
        "mode": "lightgcn",
        "kind": "cmgdr",
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
        "kind": "cmgdr",
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
        "kind": "cmgdr",
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
    {
        "name": "FREEDOM",
        "kind": "freedom",
        "epochs": 40,
        "extra": {"k_neighbors": 10},
    },
]


def main():
    seed = 42
    cfg = load_config(BEAUTY_CFG_PATH)
    print(f"Loaded Beauty config from {BEAUTY_CFG_PATH}")
    results = []
    for spec in SPECS:
        t0 = time.time()
        if spec["kind"] == "cmgdr":
            r = run_cmgdr_experiment(
                config=cfg, mode=spec["mode"], seed=seed,
                suffix=f"{spec['name']}_beauty", num_epochs=spec["epochs"],
                overrides=spec.get("overrides"),
            )
        else:
            r = run_baseline_experiment(
                FREEDOMModel, f"{spec['name']}_beauty", cfg, seed,
                num_epochs=spec["epochs"], extra_kwargs=spec.get("extra"),
            )
        r["time_sec"] = time.time() - t0
        r["seed"] = seed
        r["model"] = spec["name"]
        r["dataset"] = "Amazon_Beauty"
        results.append(r)
        with open(LOG, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
        print(f">>> {spec['name']}: R@10={r.get('Recall@10', 0):.4f} N@10={r.get('NDCG@10', 0):.4f} "
              f"Probe_c={r.get('probe_causal', r.get('probe_item_emb', 0)):.3f} "
              f"CFShift={r.get('cf_score_shift', 0):.4f}")
    save_json(NEW / "cross_domain_beauty_raw.json", results)
    print(f"\nSaved: {NEW / 'cross_domain_beauty_raw.json'}")


if __name__ == "__main__":
    main()
