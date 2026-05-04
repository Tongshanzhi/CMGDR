from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .eval import run_evaluation
from .train import run_training
from .utils import PACKAGE_ROOT, clone_config, load_config, output_paths, resolve_path


def _evaluate_and_collect(
    config: dict[str, Any],
    seed: int,
    mode: str,
    suffix: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_summary = run_training(
        config_or_path=config,
        seed=seed,
        override_mode=mode,
        suffix=suffix,
        save_outputs=True,
    )
    eval_result = run_evaluation(
        config_or_path=config,
        seed=seed,
        split_name="test",
        checkpoint_path=train_summary["checkpoint_path"],
        override_mode=mode,
        suffix=suffix,
        save_outputs=True,
    )
    row = {
        "mode": mode,
        "seed": seed,
        **{k: v for k, v in train_summary.items() if isinstance(v, (int, float, str))},
        **{k: v for k, v in eval_result["summary"].items() if isinstance(v, (int, float, str))},
    }
    return row, eval_result["bias_table"]


def run_ablations(
    config_or_path: dict[str, Any] | str | Path,
    seed: int,
    robustness_data_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    run_rows: list[dict[str, Any]] = []
    bias_tables: list[pd.DataFrame] = []

    for mode in config.get("ablation_modes", []):
        row, bias_table = _evaluate_and_collect(config=clone_config(config), seed=seed, mode=mode)
        run_rows.append(row)
        tagged_bias = bias_table.copy()
        tagged_bias.insert(0, "mode", mode)
        bias_tables.append(tagged_bias)

    ablation_table = pd.DataFrame(run_rows)
    bias_table = pd.concat(bias_tables, ignore_index=True) if bias_tables else pd.DataFrame()

    main_modes = ["lightgcn", "visual_concat", "full"]
    main_table = ablation_table[ablation_table["mode"].isin(main_modes)].reset_index(drop=True)

    sensitivity_rows: list[dict[str, Any]] = []
    for num_clusters in config.get("sensitivity", {}).get("visual_clusters", []):
        sensitivity_cfg = clone_config(config)
        sensitivity_cfg["num_visual_clusters"] = int(num_clusters)
        sensitivity_cfg["visual_cluster_path"] = ""
        row, _ = _evaluate_and_collect(
            config=sensitivity_cfg,
            seed=seed,
            mode="full",
            suffix=f"k{num_clusters}",
        )
        row["sensitivity_type"] = "num_visual_clusters"
        row["sensitivity_value"] = int(num_clusters)
        sensitivity_rows.append(row)

    for profile_name, profile_weights in config.get("sensitivity", {}).get("loss_profiles", {}).items():
        sensitivity_cfg = clone_config(config)
        sensitivity_cfg["loss_weights"] = profile_weights
        row, _ = _evaluate_and_collect(
            config=sensitivity_cfg,
            seed=seed,
            mode="full",
            suffix=f"loss_{profile_name}",
        )
        row["sensitivity_type"] = "loss_profile"
        row["sensitivity_value"] = profile_name
        sensitivity_rows.append(row)

    if robustness_data_root is not None:
        robustness_cfg = clone_config(config)
        robustness_cfg["data_root"] = str(resolve_path(PACKAGE_ROOT, robustness_data_root))
        robustness_cfg["category"] = "Clothing_Shoes_and_Jewelry"
        row, _ = _evaluate_and_collect(
            config=robustness_cfg,
            seed=seed,
            mode="full",
            suffix="clothing_robustness",
        )
        row["sensitivity_type"] = "category_robustness"
        row["sensitivity_value"] = "Clothing_Shoes_and_Jewelry"
        sensitivity_rows.append(row)

    sensitivity_table = pd.DataFrame(sensitivity_rows)
    paths = output_paths(config, seed=seed, mode="full", suffix="ablation_bundle")
    main_table.to_csv(paths["metrics"].with_name("main_table.csv"), index=False)
    ablation_table.to_csv(paths["metrics"].with_name("ablation_table.csv"), index=False)
    bias_table.to_csv(paths["metrics"].with_name("bias_table.csv"), index=False)
    sensitivity_table.to_csv(paths["metrics"].with_name("sensitivity_table.csv"), index=False)

    return {
        "main_table": main_table,
        "ablation_table": ablation_table,
        "bias_table": bias_table,
        "sensitivity_table": sensitivity_table,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CMGDR ablations and sensitivity sweeps")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config" / "model.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robustness-data-root", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    tables = run_ablations(
        config_or_path=args.config,
        seed=args.seed,
        robustness_data_root=args.robustness_data_root,
    )
    print({name: len(frame) for name, frame in tables.items()})
