"""Section 7.7 — Efficiency report.

Combine wall-clock time and trainable parameter count from
  shared_data/evaluation/baseline_results.json (9 published baselines)
  shared_data/evaluation/full_evaluation_results.json (5 CMGDR variants)
  new_results/multiseed_raw.json (3-seed CMGDR/FREEDOM/MM-LightGCN runs)
into one summary table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module_B"))

from src.utils import ensure_dir, save_json

SHARED = ROOT / "shared_data"
NEW = ensure_dir(ROOT / "new_results")


def _load_json(path):
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def parameter_count_for_method(name: str, ckpt_dir: Path):
    """Pull n_params from a saved checkpoint if available."""
    candidates = [
        ckpt_dir / f"{name}_seed42.pt",
        ckpt_dir / f"{name}_seed42_seed42.pt",
    ]
    for c in candidates:
        if c.exists():
            try:
                state = torch.load(c, map_location="cpu", weights_only=False)
            except Exception:
                continue
            sd = state.get("model_state_dict", state)
            if not isinstance(sd, dict):
                continue
            return sum(int(np.prod(t.shape)) for t in sd.values() if hasattr(t, "shape"))
    return None


def main():
    baseline = _load_json(SHARED / "evaluation" / "baseline_results.json")
    full = _load_json(SHARED / "evaluation" / "full_evaluation_results.json")
    multi = _load_json(NEW / "multiseed_raw.json")

    rows = []
    ckpt_dir = SHARED / "model_outputs" / "checkpoints"

    # 9 published baselines
    for r in baseline:
        rows.append({
            "method": r["model"],
            "category": "baseline-published",
            "Recall@10": r["Recall@10"],
            "NDCG@10": r["NDCG@10"],
            "n_params": r.get("n_params") or parameter_count_for_method(r["model"], ckpt_dir),
            "wallclock_train_sec": r.get("time_sec"),
            "embedding_dim": r.get("embedding_dim", 256),
            "seed": 42,
        })

    # 5 CMGDR variants from existing run
    for r in full:
        rows.append({
            "method": r["suffix"],
            "category": "CMGDR-variant",
            "Recall@10": r["Recall@10"],
            "NDCG@10": r["NDCG@10"],
            "n_params": parameter_count_for_method(r["suffix"], ckpt_dir),
            "wallclock_train_sec": r.get("time_sec"),
            "embedding_dim": r.get("config", {}).get("embedding_dim")
                if isinstance(r.get("config"), dict) else None,
            "seed": 42,
        })

    # Multi-seed runs (one row per (model, seed))
    for r in multi:
        rows.append({
            "method": f"{r.get('model','?')}_seed{r.get('seed','?')}",
            "category": "multi-seed",
            "Recall@10": r.get("Recall@10"),
            "NDCG@10": r.get("NDCG@10"),
            "n_params": r.get("n_params"),
            "wallclock_train_sec": r.get("time_sec"),
            "embedding_dim": None,
            "seed": r.get("seed"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(NEW / "efficiency_report.csv", index=False)
    save_json(NEW / "efficiency_report.json", rows)
    # Pretty summary
    base_only = df[df["category"] != "multi-seed"].copy()
    if not base_only.empty:
        base_only_sorted = base_only.sort_values("Recall@10", ascending=False)
        print(f"\n{'Method':<20} {'R@10':>7} {'N@10':>7} {'Params':>12} {'Train sec':>10}")
        print("-" * 60)
        for _, r in base_only_sorted.iterrows():
            np_str = f"{int(r['n_params']):,}" if pd.notna(r["n_params"]) else "?"
            ws_str = f"{r['wallclock_train_sec']:.0f}" if pd.notna(r["wallclock_train_sec"]) else "?"
            print(f"{r['method']:<20} {r['Recall@10']:7.4f} {r['NDCG@10']:7.4f} {np_str:>12} {ws_str:>10}")
    print(f"\nSaved: {NEW / 'efficiency_report.csv'}")


if __name__ == "__main__":
    main()
