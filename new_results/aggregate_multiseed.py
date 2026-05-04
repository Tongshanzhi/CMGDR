"""Aggregate Section 7.1 multi-seed runs into mean ± std table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "module_B"))

from src.utils import ensure_dir, save_json

NEW = ensure_dir(ROOT / "new_results")


METRICS = [
    "Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "HR@10", "MRR",
    "exposure_gap@10", "calibration_gap@10",
    "probe_graph", "probe_causal", "cf_score_shift",
]


def main():
    raw_path = NEW / "multiseed_raw.json"
    if not raw_path.exists():
        print(f"Missing {raw_path}. Run run_multiseed.py first.")
        return

    raw = json.loads(raw_path.read_text())
    rows = []
    for r in raw:
        model = r.get("model", r.get("suffix", "?"))
        seed = r.get("seed", "?")
        row = {"model": model, "seed": seed}
        for k in METRICS:
            if k in r and isinstance(r[k], (int, float)):
                row[k] = r[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\nRaw rows: {len(df)}")

    # Aggregate by model
    grouped = df.groupby("model")
    agg_rows = []
    for model, g in grouped:
        out = {"model": model, "n_seeds": len(g)}
        for k in METRICS:
            if k in g.columns and g[k].notna().any():
                out[f"{k}_mean"] = float(g[k].mean())
                out[f"{k}_std"] = float(g[k].std(ddof=1)) if len(g) > 1 else 0.0
        agg_rows.append(out)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(NEW / "multiseed_summary.csv", index=False)
    save_json(NEW / "multiseed_summary.json", agg_rows)

    # Pretty print
    print(f"\n{'model':<18} {'seeds':>5} {'R@10 mean':>11} {'R@10 std':>9} "
          f"{'N@10 mean':>11} {'N@10 std':>9} {'Probe':>8} {'CFShift':>8}")
    print("-" * 90)
    for r in agg_rows:
        print(f"{r['model']:<18} {r['n_seeds']:>5} "
              f"{r.get('Recall@10_mean', 0):>11.4f} {r.get('Recall@10_std', 0):>9.4f} "
              f"{r.get('NDCG@10_mean', 0):>11.4f} {r.get('NDCG@10_std', 0):>9.4f} "
              f"{r.get('probe_causal_mean', r.get('probe_item_emb_mean', 0)):>8.3f} "
              f"{r.get('cf_score_shift_mean', 0):>8.4f}")

    print(f"\nSaved: {NEW / 'multiseed_summary.csv'}")


if __name__ == "__main__":
    main()
