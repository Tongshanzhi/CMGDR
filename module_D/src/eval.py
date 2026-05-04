from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow importing from Module B (data, features, utils, models)
_CMGDR_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_CMGDR_ROOT / "module_B"), str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import build_symmetric_edge_index, load_data_bundle
from src.features import prepare_visual_artifacts
from metrics import (
    aggregate_metric_dicts,
    cluster_calibration_gap,
    cluster_exposure_gap,
    cluster_probe_accuracy,
    ranking_metrics_for_user,
)
from src.utils import PACKAGE_ROOT, load_config, output_paths, resolve_path, save_json, topk_list


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for evaluation. Install the modeling dependencies first."
        ) from exc
    return torch


def _choose_device(torch_module, config: dict[str, Any]):
    requested = str(config.get("device", "auto")).lower()
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda":
        return torch_module.device("cuda")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _prepare_runtime(
    config: dict[str, Any],
    seed: int,
    mode: str | None = None,
):
    torch = _require_torch()
    data_root = resolve_path(PACKAGE_ROOT, config["data_root"])
    if data_root is None:
        raise ValueError("config.data_root is required")
    bundle = load_data_bundle(data_root)
    visual_artifacts = prepare_visual_artifacts(config, PACKAGE_ROOT, bundle.n_items, seed)
    device = _choose_device(torch, config)

    edge_index = torch.from_numpy(build_symmetric_edge_index(bundle.edges_train))
    from src.models.lightgcn_backbone import build_normalized_adjacency

    norm_adj = build_normalized_adjacency(edge_index=edge_index, n_nodes=bundle.n_nodes, device=device)
    visual_features = torch.tensor(visual_artifacts.features, dtype=torch.float32, device=device)
    visual_clusters = torch.tensor(visual_artifacts.clusters, dtype=torch.long, device=device)
    cluster_prototypes = torch.tensor(visual_artifacts.prototypes, dtype=torch.float32, device=device)

    from src.models.causal_debias import CMGDRModel

    # Load text features if available
    import numpy as np
    text_features_t = None
    text_dim = 0
    text_path = resolve_path(Path(data_root), config.get("text_feature_path", ""))
    if text_path is not None and text_path.exists():
        text_np = np.load(text_path).astype(np.float32)
        text_dim = text_np.shape[1]
        text_features_t = torch.tensor(text_np, dtype=torch.float32, device=device)

    # Load item-item graph if available
    import pandas as pd_eval
    item_item_adj = None
    use_item_item = config.get("use_item_item_graph", False)
    ii_path = resolve_path(Path(data_root), config.get("item_item_graph_path", ""))
    if use_item_item and ii_path is not None and ii_path.exists():
        ii_df = pd_eval.read_parquet(ii_path)
        rows = np.concatenate([ii_df["item_idx_a"].values, ii_df["item_idx_b"].values])
        cols = np.concatenate([ii_df["item_idx_b"].values, ii_df["item_idx_a"].values])
        vals = np.ones(len(rows), dtype=np.float32)
        from scipy import sparse as sp
        ii_coo = sp.coo_matrix((vals, (rows, cols)), shape=(bundle.n_items, bundle.n_items))
        ii_coo.sum_duplicates()
        deg = np.array(ii_coo.sum(axis=1)).flatten().clip(min=1.0)
        deg_inv_sqrt = np.power(deg, -0.5)
        ii_coo = sp.diags(deg_inv_sqrt) @ ii_coo @ sp.diags(deg_inv_sqrt)
        ii_coo = ii_coo.tocoo()
        indices = torch.tensor(np.stack([ii_coo.row, ii_coo.col]), dtype=torch.long, device=device)
        values = torch.tensor(ii_coo.data, dtype=torch.float32, device=device)
        item_item_adj = torch.sparse_coo_tensor(indices, values, (bundle.n_items, bundle.n_items)).coalesce()

    resolved_mode = mode or config["debias_mode"]
    model = CMGDRModel(
        n_users=bundle.n_users,
        n_items=bundle.n_items,
        visual_dim=visual_artifacts.features.shape[1],
        num_clusters=int(visual_artifacts.summary["num_clusters"]),
        embedding_dim=int(config["embedding_dim"]),
        num_layers=int(config["num_layers"]),
        text_dim=text_dim,
        use_item_item_graph=use_item_item,
    ).to(device)

    return {
        "torch": torch,
        "device": device,
        "bundle": bundle,
        "visual_artifacts": visual_artifacts,
        "norm_adj": norm_adj,
        "visual_features": visual_features,
        "visual_clusters": visual_clusters,
        "cluster_prototypes": cluster_prototypes,
        "text_features": text_features_t,
        "item_item_adj": item_item_adj,
        "model": model,
        "mode": resolved_mode,
    }


def _default_checkpoint(config: dict[str, Any], seed: int, mode: str | None = None, suffix: str | None = None) -> Path:
    return output_paths(config, seed=seed, mode=mode, suffix=suffix)["checkpoint"]


def load_checkpoint(
    model,
    checkpoint_path: str | Path,
    device,
):
    torch = _require_torch()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    return ckpt


def evaluate_model(
    model,
    norm_adj,
    visual_features,
    visual_clusters,
    cluster_prototypes,
    bundle,
    config: dict[str, Any],
    split_name: str,
    device,
    mode: str,
    text_features=None,
    item_item_adj=None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    torch = _require_torch()
    topk = topk_list(config)
    max_topk = max(topk)
    user_batch_size = int(config.get("candidate_batch_size", 256))
    split = bundle.eval_splits[split_name]

    model.eval()
    with torch.no_grad():
        outputs = model.encode_all(
            norm_adj=norm_adj,
            visual_features=visual_features,
            visual_clusters=visual_clusters,
            cluster_prototypes=cluster_prototypes,
            mode=mode,
            grl_lambda=0.0,
            text_features=text_features,
            item_item_adj=item_item_adj,
        )

    user_embeddings = outputs["user_embeddings"]
    item_causal_embeddings = outputs["item_causal_embeddings"]
    item_total_embeddings = outputs["item_total_embeddings"]
    counterfactual_causal_embeddings = outputs["counterfactual_causal_embeddings"]
    counterfactual_total_embeddings = outputs["counterfactual_total_embeddings"]

    recommendations_by_user: dict[int, np.ndarray] = {}
    metrics_per_user: list[dict[str, float]] = []
    causal_shift: list[float] = []
    total_shift: list[float] = []

    for start in range(0, len(split.user_ids), user_batch_size):
        batch_users = split.user_ids[start : start + user_batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
        causal_scores = torch.matmul(user_embeddings[user_tensor], item_causal_embeddings.T)
        total_scores = torch.matmul(user_embeddings[user_tensor], item_total_embeddings.T)
        cf_causal_scores = torch.matmul(user_embeddings[user_tensor], counterfactual_causal_embeddings.T)
        cf_total_scores = torch.matmul(user_embeddings[user_tensor], counterfactual_total_embeddings.T)

        for local_idx, user_idx in enumerate(batch_users.tolist()):
            mask_items = split.mask_by_user[user_idx]
            causal_row = causal_scores[local_idx].clone()
            total_row = total_scores[local_idx].clone()
            if len(mask_items) > 0:
                mask_tensor = torch.tensor(mask_items, dtype=torch.long, device=device)
                causal_row[mask_tensor] = -1e9
                total_row[mask_tensor] = -1e9

            target_items = split.targets_by_user[user_idx]
            target_tensor = torch.tensor(target_items, dtype=torch.long, device=device)
            best_target_score = torch.max(causal_row[target_tensor])
            reciprocal_rank = 1.0 / float(1 + torch.sum(causal_row > best_target_score).item())

            recommended = torch.topk(causal_row, k=max_topk).indices.detach().cpu().numpy()
            user_metrics = ranking_metrics_for_user(
                recommended=recommended,
                targets=target_items,
                topk=topk,
            )
            user_metrics["MRR"] = reciprocal_rank
            recommendations_by_user[user_idx] = recommended
            metrics_per_user.append(user_metrics)
            causal_shift.append(
                float(torch.mean(torch.abs(causal_scores[local_idx] - cf_causal_scores[local_idx])).item())
            )
            total_shift.append(
                float(torch.mean(torch.abs(total_scores[local_idx] - cf_total_scores[local_idx])).item())
            )

    summary = aggregate_metric_dicts(metrics_per_user)
    exposure_gap, exposure_frame = cluster_exposure_gap(
        recommendations_by_user=recommendations_by_user,
        visual_clusters=visual_clusters.detach().cpu().numpy(),
        topk=max_topk,
    )
    calibration_gap, calibration_frame = cluster_calibration_gap(
        recommendations_by_user=recommendations_by_user,
        targets_by_user=split.targets_by_user,
        visual_clusters=visual_clusters.detach().cpu().numpy(),
        topk=max_topk,
    )
    bias_table = exposure_frame.merge(
        calibration_frame.rename(columns={"exposure_share": "exposure_share_vs_target"}),
        on="cluster_id",
        how="outer",
    )

    graph_probe = cluster_probe_accuracy(
        embeddings=outputs["item_graph_embeddings"].detach().cpu().numpy(),
        visual_clusters=visual_clusters.detach().cpu().numpy(),
        test_ratio=float(config.get("cluster_probe_test_ratio", 0.2)),
        random_state=42,
        max_iter=int(config.get("cluster_probe_max_iter", 300)),
    )
    causal_probe = cluster_probe_accuracy(
        embeddings=outputs["item_causal_embeddings"].detach().cpu().numpy(),
        visual_clusters=visual_clusters.detach().cpu().numpy(),
        test_ratio=float(config.get("cluster_probe_test_ratio", 0.2)),
        random_state=42,
        max_iter=int(config.get("cluster_probe_max_iter", 300)),
    )

    summary.update(
        {
            "split": split_name,
            "mode": mode,
            "visual_cluster_exposure_gap": float(exposure_gap),
            "cluster_wise_calibration_gap": float(calibration_gap),
            "counterfactual_score_shift_causal": float(np.mean(causal_shift) if causal_shift else 0.0),
            "counterfactual_score_shift_total": float(np.mean(total_shift) if total_shift else 0.0),
            "cluster_probe_graph": float(graph_probe),
            "cluster_probe_causal": float(causal_probe),
            "n_eval_users": int(len(split.user_ids)),
        }
    )
    return summary, bias_table


def plot_counterfactual_shift(summary: dict[str, Any], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    labels = ["s_total", "s_causal"]
    values = [
        float(summary["counterfactual_score_shift_total"]),
        float(summary["counterfactual_score_shift_causal"]),
    ]
    colors = ["#c45b2d", "#17766f"]
    ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylabel("Mean absolute counterfactual score shift")
    ax.set_title(f"Counterfactual stability ({summary['mode']})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def run_evaluation(
    config_or_path: dict[str, Any] | str | Path,
    seed: int,
    split_name: str = "test",
    checkpoint_path: str | Path | None = None,
    override_mode: str | None = None,
    suffix: str | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    runtime = _prepare_runtime(config=config, seed=seed, mode=override_mode)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else _default_checkpoint(
        config, seed=seed, mode=override_mode, suffix=suffix
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    load_checkpoint(runtime["model"], checkpoint, runtime["device"])

    summary, bias_table = evaluate_model(
        model=runtime["model"],
        norm_adj=runtime["norm_adj"],
        visual_features=runtime["visual_features"],
        visual_clusters=runtime["visual_clusters"],
        cluster_prototypes=runtime["cluster_prototypes"],
        bundle=runtime["bundle"],
        config=config,
        split_name=split_name,
        device=runtime["device"],
        mode=runtime["mode"],
        text_features=runtime.get("text_features"),
        item_item_adj=runtime.get("item_item_adj"),
    )
    summary["checkpoint_path"] = str(checkpoint)

    if save_outputs:
        paths = output_paths(config, seed=seed, mode=runtime["mode"], suffix=suffix)
        save_json(paths["metrics"], summary)
        bias_table.to_csv(paths["bias"], index=False)
        plot_counterfactual_shift(summary, paths["shift_figure"])

    return {"summary": summary, "bias_table": bias_table}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CMGDR checkpoints on valid/test splits")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config" / "model.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    result = run_evaluation(
        config_or_path=args.config,
        seed=args.seed,
        split_name=args.split,
        checkpoint_path=args.checkpoint,
        override_mode=args.mode,
        suffix=args.suffix,
        save_outputs=True,
    )
    print(result["summary"])
