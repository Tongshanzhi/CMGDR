from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .data import PairwiseTrainDataset, load_data_bundle
from .eval import evaluate_model
from .features import prepare_visual_artifacts
from .losses import (
    adversarial_loss,
    bpr_loss,
    counterfactual_consistency_loss,
    cross_modal_contrastive_loss,
    orthogonality_loss,
    residual_loss,
    text_consistency_loss,
)
from .utils import (
    PACKAGE_ROOT,
    active_loss_weights,
    append_jsonl,
    count_trainable_parameters,
    load_config,
    output_paths,
    resolve_path,
    save_json,
    set_seed,
)


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "torch is required for training. Install the modeling dependencies first."
        ) from exc
    return torch, DataLoader


def _choose_device(torch_module, config: dict[str, Any]):
    requested = str(config.get("device", "auto")).lower()
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda":
        return torch_module.device("cuda")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _prepare_runtime(config: dict[str, Any], seed: int, mode: str | None = None):
    from .data import build_symmetric_edge_index
    from .models.causal_debias import CMGDRModel
    from .models.lightgcn_backbone import build_normalized_adjacency

    torch, DataLoader = _require_torch()
    data_root = resolve_path(PACKAGE_ROOT, config["data_root"])
    if data_root is None:
        raise ValueError("config.data_root is required")
    bundle = load_data_bundle(data_root)
    visual_artifacts = prepare_visual_artifacts(config, PACKAGE_ROOT, bundle.n_items, seed)
    device = _choose_device(torch, config)

    edge_index = torch.from_numpy(build_symmetric_edge_index(bundle.edges_train))
    norm_adj = build_normalized_adjacency(edge_index=edge_index, n_nodes=bundle.n_nodes, device=device)
    visual_features = torch.tensor(visual_artifacts.features, dtype=torch.float32, device=device)
    visual_clusters = torch.tensor(visual_artifacts.clusters, dtype=torch.long, device=device)
    cluster_prototypes = torch.tensor(visual_artifacts.prototypes, dtype=torch.float32, device=device)

    # Load text features if available
    import numpy as np
    text_features_t = None
    text_dim = 0
    text_path = resolve_path(Path(data_root), config.get("text_feature_path", ""))
    if text_path is not None and text_path.exists():
        text_np = np.load(text_path).astype(np.float32)
        assert text_np.shape[0] == bundle.n_items
        text_dim = text_np.shape[1]
        text_features_t = torch.tensor(text_np, dtype=torch.float32, device=device)
        print(f"Loaded text features: {text_np.shape}")

    # Load item-item copurchase graph if available
    import pandas as pd
    item_item_adj = None
    use_item_item = config.get("use_item_item_graph", False)
    ii_path = resolve_path(Path(data_root), config.get("item_item_graph_path", ""))
    if use_item_item and ii_path is not None and ii_path.exists():
        ii_df = pd.read_parquet(ii_path)
        rows = np.concatenate([ii_df["item_idx_a"].values, ii_df["item_idx_b"].values])
        cols = np.concatenate([ii_df["item_idx_b"].values, ii_df["item_idx_a"].values])
        vals = np.ones(len(rows), dtype=np.float32)
        from scipy import sparse as sp
        ii_coo = sp.coo_matrix((vals, (rows, cols)), shape=(bundle.n_items, bundle.n_items))
        ii_coo.sum_duplicates()
        # Symmetric normalize: D^{-1/2} A D^{-1/2}
        deg = np.array(ii_coo.sum(axis=1)).flatten().clip(min=1.0)
        deg_inv_sqrt = np.power(deg, -0.5)
        ii_coo = sp.diags(deg_inv_sqrt) @ ii_coo @ sp.diags(deg_inv_sqrt)
        ii_coo = ii_coo.tocoo()
        indices = torch.tensor(np.stack([ii_coo.row, ii_coo.col]), dtype=torch.long, device=device)
        values = torch.tensor(ii_coo.data, dtype=torch.float32, device=device)
        item_item_adj = torch.sparse_coo_tensor(indices, values, (bundle.n_items, bundle.n_items)).coalesce()
        print(f"Loaded item-item graph: {ii_df.shape[0]} edges -> {item_item_adj._nnz()} nnz")

    # Build per-item average review rating (normalized to [0,1]) for text consistency loss
    review_ratings_t = None
    if config.get("loss_weights", {}).get("text_consistency", 0.0) > 0:
        train_inter = bundle.interactions[bundle.interactions["split"] == "train"]
        avg_rating = train_inter.groupby("item_idx")["overall"].mean()
        rating_arr = np.full(bundle.n_items, -1.0, dtype=np.float32)
        for idx, val in avg_rating.items():
            rating_arr[int(idx)] = (float(val) - 1.0) / 4.0  # map [1,5] -> [0,1]
        review_ratings_t = torch.tensor(rating_arr, dtype=torch.float32, device=device)

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
    dataset = PairwiseTrainDataset(
        interactions=bundle.interactions,
        n_items=bundle.n_items,
        user_pos_all=bundle.user_pos_all,
        negatives_per_pos=int(config.get("negatives_per_pos", 1)),
        seed=seed,
        visual_clusters=visual_artifacts.clusters,
        stratified_sampling=bool(config.get("stratified_sampling", False)),
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True)
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
        "review_ratings": review_ratings_t,
        "model": model,
        "loader": loader,
        "mode": resolved_mode,
    }


def _compute_objective(
    outputs: dict[str, Any],
    visual_clusters,
    weight_map: dict[str, float],
    review_ratings=None,
    pos_item_indices=None,
):
    loss_rank = bpr_loss(outputs["pos_causal_scores"], outputs["neg_causal_scores"])
    total_loss = loss_rank
    metrics = {"loss_rank": float(loss_rank.item())}

    if weight_map["residual"] > 0:
        loss_res = residual_loss(
            outputs["item_graph_embeddings"],
            outputs["item_causal_embeddings"],
            outputs["item_bias_embeddings"],
        )
        total_loss = total_loss + weight_map["residual"] * loss_res
        metrics["loss_residual"] = float(loss_res.item())
    else:
        metrics["loss_residual"] = 0.0

    if weight_map["adversarial"] > 0 and outputs["cluster_logits"] is not None:
        loss_adv = adversarial_loss(outputs["cluster_logits"], visual_clusters)
        total_loss = total_loss + weight_map["adversarial"] * loss_adv
        metrics["loss_adversarial"] = float(loss_adv.item())
    else:
        metrics["loss_adversarial"] = 0.0

    if weight_map["counterfactual"] > 0:
        loss_cf = 0.5 * (
            counterfactual_consistency_loss(
                outputs["pos_causal_scores"], outputs["pos_causal_cf_scores"]
            )
            + counterfactual_consistency_loss(
                outputs["neg_causal_scores"], outputs["neg_causal_cf_scores"]
            )
        )
        total_loss = total_loss + weight_map["counterfactual"] * loss_cf
        metrics["loss_counterfactual"] = float(loss_cf.item())
    else:
        metrics["loss_counterfactual"] = 0.0

    if weight_map["orthogonality"] > 0:
        loss_orth = orthogonality_loss(
            outputs["item_causal_embeddings"], outputs["item_bias_embeddings"]
        )
        total_loss = total_loss + weight_map["orthogonality"] * loss_orth
        metrics["loss_orthogonality"] = float(loss_orth.item())
    else:
        metrics["loss_orthogonality"] = 0.0

    # Cross-modal contrastive loss (visual <-> text)
    if weight_map.get("contrastive", 0.0) > 0 and outputs.get("visual_latent") is not None and outputs.get("text_latent") is not None:
        # Sample a subset of items to avoid OOM on full (n_items x n_items) logits
        n_items = outputs["visual_latent"].size(0)
        sample_size = min(2048, n_items)
        idx = outputs["visual_latent"].new_zeros(sample_size, dtype=outputs["visual_latent"].dtype)
        import torch as _t
        sample_idx = _t.randperm(n_items, device=outputs["visual_latent"].device)[:sample_size]
        loss_cl = cross_modal_contrastive_loss(
            outputs["visual_latent"][sample_idx],
            outputs["text_latent"][sample_idx],
            temperature=float(weight_map.get("contrastive_temperature", 0.2)),
        )
        total_loss = total_loss + weight_map["contrastive"] * loss_cl
        metrics["loss_contrastive"] = float(loss_cl.item())
    else:
        metrics["loss_contrastive"] = 0.0

    # Text consistency loss (predicted score vs review rating)
    if weight_map.get("text_consistency", 0.0) > 0 and review_ratings is not None and pos_item_indices is not None:
        item_ratings = review_ratings[pos_item_indices]
        valid_mask = item_ratings >= 0  # -1 means no review
        if valid_mask.any():
            loss_tc = text_consistency_loss(
                outputs["pos_causal_scores"][valid_mask],
                item_ratings[valid_mask],
            )
            total_loss = total_loss + weight_map["text_consistency"] * loss_tc
            metrics["loss_text_consistency"] = float(loss_tc.item())
        else:
            metrics["loss_text_consistency"] = 0.0
    else:
        metrics["loss_text_consistency"] = 0.0

    metrics["loss_total"] = float(total_loss.item())
    return total_loss, metrics


def run_training(
    config_or_path: dict[str, Any] | str | Path,
    seed: int,
    override_mode: str | None = None,
    suffix: str | None = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    set_seed(seed)
    runtime = _prepare_runtime(config=config, seed=seed, mode=override_mode)
    torch = runtime["torch"]
    model = runtime["model"]
    mode = runtime["mode"]
    weight_map = active_loss_weights(config, mode=mode)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    paths = output_paths(config, seed=seed, mode=mode, suffix=suffix)
    if paths["history"].exists():
        paths["history"].unlink()

    best_valid = float("-inf")
    best_summary: dict[str, Any] | None = None

    for epoch in range(1, int(config["num_epochs"]) + 1):
        model.train()
        epoch_losses: list[dict[str, float]] = []
        for batch in runtime["loader"]:
            batch_user = batch["user_idx"].to(runtime["device"])
            batch_pos = batch["pos_item_idx"].to(runtime["device"])
            batch_neg = batch["neg_item_idx"].to(runtime["device"])
            outputs = model(
                user_indices=batch_user,
                pos_item_indices=batch_pos,
                neg_item_indices=batch_neg,
                norm_adj=runtime["norm_adj"],
                visual_features=runtime["visual_features"],
                visual_clusters=runtime["visual_clusters"],
                cluster_prototypes=runtime["cluster_prototypes"],
                mode=mode,
                grl_lambda=weight_map["adversarial"],
                text_features=runtime.get("text_features"),
                item_item_adj=runtime.get("item_item_adj"),
            )
            loss, metrics = _compute_objective(
                outputs=outputs,
                visual_clusters=runtime["visual_clusters"],
                weight_map=weight_map,
                review_ratings=runtime.get("review_ratings"),
                pos_item_indices=batch_pos,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(metrics)

        averaged = {
            key: float(sum(loss_row[key] for loss_row in epoch_losses) / max(len(epoch_losses), 1))
            for key in epoch_losses[0].keys()
        }
        history_row: dict[str, Any] = {"epoch": epoch, "mode": mode, **averaged}

        if epoch % int(config.get("eval_every", 1)) == 0:
            valid_summary, _ = evaluate_model(
                model=model,
                norm_adj=runtime["norm_adj"],
                visual_features=runtime["visual_features"],
                visual_clusters=runtime["visual_clusters"],
                cluster_prototypes=runtime["cluster_prototypes"],
                bundle=runtime["bundle"],
                config=config,
                split_name="valid",
                text_features=runtime.get("text_features"),
                item_item_adj=runtime.get("item_item_adj"),
                device=runtime["device"],
                mode=mode,
            )
            history_row.update({f"valid_{k}": v for k, v in valid_summary.items() if isinstance(v, (int, float))})
            current_valid = float(valid_summary.get("Recall@10", 0.0))
            if current_valid >= best_valid:
                best_valid = current_valid
                best_summary = valid_summary
                if save_outputs:
                    torch.save(
                        {
                            "epoch": epoch,
                            "mode": mode,
                            "seed": seed,
                            "config": config,
                            "model_state_dict": model.state_dict(),
                            "best_valid_summary": valid_summary,
                        },
                        paths["checkpoint"],
                    )
        append_jsonl(paths["history"], history_row)

    if save_outputs and not paths["checkpoint"].exists():
        torch.save(
            {
                "epoch": int(config["num_epochs"]),
                "mode": mode,
                "seed": seed,
                "config": config,
                "model_state_dict": model.state_dict(),
                "best_valid_summary": best_summary or {},
            },
            paths["checkpoint"],
        )

    train_summary = {
        "mode": mode,
        "seed": seed,
        "checkpoint_path": str(paths["checkpoint"]),
        "best_valid_Recall@10": float(best_valid if best_valid != float("-inf") else 0.0),
        "parameter_count": count_trainable_parameters(model),
        "loss_weights": weight_map,
    }
    if best_summary is not None:
        train_summary["best_valid_summary"] = best_summary
    if save_outputs:
        save_json(paths["metrics"].with_name(paths["metrics"].stem + "_train_summary.json"), train_summary)
    return train_summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the visual-first CMGDR model")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config" / "model.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    summary = run_training(
        config_or_path=args.config,
        seed=args.seed,
        override_mode=args.mode,
        suffix=args.suffix,
        save_outputs=True,
    )
    print(summary)
