"""Diagnostic v2: critical test — is visual signal helpful or harmful in Beauty?

If Visual-Concat (visual feature + graph embedding, no debias loss) BEATS
MM-LightGCN on Beauty -> visual is informative -> CMGDR's debias is wasting
useful signal. If Visual-Concat LOSES to MM-LightGCN like in Sports -> visual
is still a confounder and the gap must be explained differently.

Also test the "default" loss profile that includes contrastive + text_consistency,
which the headline run had disabled.
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
LOG = NEW / "beauty_diagnose_v2_log.jsonl"
LOG.unlink(missing_ok=True)


SPECS = [
    {
        "name": "Visual-Concat_beauty",
        "mode": "visual_concat",
        "seed": 42,
        "overrides": {
            "use_item_item_graph": True,
            "embedding_dim": 256,
            "learning_rate": 0.002,
            "loss_weights": {
                "residual": 0, "adversarial": 0, "counterfactual": 0,
                "orthogonality": 0, "contrastive": 0,
                "contrastive_temperature": 0.2, "text_consistency": 0,
            },
        },
    },
    {
        "name": "CMGDR-Full_default_profile",
        "mode": "full",
        "seed": 42,
        "overrides": {
            "use_item_item_graph": True,
            "embedding_dim": 256,
            "learning_rate": 0.002,
            "loss_weights": {
                "residual": 1.0, "adversarial": 0.01, "counterfactual": 0.5,
                "orthogonality": 0.1, "contrastive": 0.1,
                "contrastive_temperature": 0.2, "text_consistency": 0.05,
            },
        },
    },
]


def main():
    cfg = load_config(BEAUTY_CFG)
    results = []
    for spec in SPECS:
        t0 = time.time()
        r = run_cmgdr_experiment(
            config=cfg, mode=spec["mode"], seed=spec["seed"],
            suffix=f"{spec['name']}_seed{spec['seed']}", num_epochs=40,
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
        cfs = r.get("cf_score_shift", 0)
        print(f">>> {spec['name']}: R@10={r.get('Recall@10', 0):.4f} "
              f"valid={r.get('best_valid_R@10', 0):.4f} "
              f"N@10={r.get('NDCG@10', 0):.4f} Probe={probe:.3f} CF={cfs:.4f}")
    save_json(NEW / "beauty_diagnose_v2_raw.json", results)
    print(f"Saved: {NEW / 'beauty_diagnose_v2_raw.json'}")


if __name__ == "__main__":
    main()
