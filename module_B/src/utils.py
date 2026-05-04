from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)


def resolve_path(base_dir: str | Path, maybe_relative: str | Path | None) -> Path | None:
    if maybe_relative in (None, ""):
        return None
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def topk_list(config: dict[str, Any]) -> list[int]:
    raw = config.get("topk", [10])
    if isinstance(raw, int):
        return [raw]
    return sorted(int(x) for x in raw)


def mode_components(mode: str) -> dict[str, bool]:
    mode = mode.lower()
    mapping = {
        "lightgcn": {"visual": False, "residual": False, "adversarial": False, "counterfactual": False},
        "visual_concat": {"visual": True, "residual": False, "adversarial": False, "counterfactual": False},
        "residual": {"visual": True, "residual": True, "adversarial": False, "counterfactual": False},
        "adversarial": {"visual": True, "residual": False, "adversarial": True, "counterfactual": False},
        "counterfactual": {"visual": True, "residual": False, "adversarial": False, "counterfactual": True},
        "residual_adv": {"visual": True, "residual": True, "adversarial": True, "counterfactual": False},
        "full": {"visual": True, "residual": True, "adversarial": True, "counterfactual": True},
    }
    if mode not in mapping:
        raise ValueError(f"Unknown debias_mode: {mode}")
    return mapping[mode]


def active_loss_weights(config: dict[str, Any], mode: str | None = None) -> dict[str, float]:
    resolved_mode = mode or config["debias_mode"]
    flags = mode_components(resolved_mode)
    raw = config.get("loss_weights", {})
    weights = {
        "residual": float(raw.get("residual", 0.0)) if flags["residual"] else 0.0,
        "adversarial": float(raw.get("adversarial", 0.0)) if flags["adversarial"] else 0.0,
        "counterfactual": float(raw.get("counterfactual", 0.0)) if flags["counterfactual"] else 0.0,
        "orthogonality": float(raw.get("orthogonality", 0.0)) if flags["visual"] else 0.0,
        "contrastive": float(raw.get("contrastive", 0.0)),
        "contrastive_temperature": float(raw.get("contrastive_temperature", 0.2)),
        "text_consistency": float(raw.get("text_consistency", 0.0)),
    }
    return weights


def run_name(config: dict[str, Any], seed: int, mode: str | None = None, suffix: str | None = None) -> str:
    resolved_mode = mode or config["debias_mode"]
    parts = [
        str(config.get("category", "dataset")),
        resolved_mode,
        f"seed{seed}",
    ]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def output_paths(config: dict[str, Any], seed: int, mode: str | None = None, suffix: str | None = None) -> dict[str, Path]:
    name = run_name(config, seed, mode=mode, suffix=suffix)
    checkpoint_dir = ensure_dir(resolve_path(PACKAGE_ROOT, config.get("checkpoint_dir", "checkpoints")))
    result_dir = ensure_dir(resolve_path(PACKAGE_ROOT, config.get("result_dir", "results")))
    log_dir = ensure_dir(resolve_path(PACKAGE_ROOT, config.get("log_dir", "logs")))
    artifact_dir = ensure_dir(resolve_path(PACKAGE_ROOT, config.get("artifact_dir", "artifacts")))
    return {
        "run_name": Path(name),
        "checkpoint": checkpoint_dir / f"{name}.pt",
        "metrics": result_dir / f"{name}_metrics.json",
        "bias": result_dir / f"{name}_bias.csv",
        "shift_figure": result_dir / f"{name}_counterfactual_shift.png",
        "history": log_dir / f"{name}.jsonl",
        "artifact_dir": artifact_dir,
    }


def count_trainable_parameters(model: Any) -> int:
    if not hasattr(model, "parameters"):
        return 0
    return int(sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False)))
