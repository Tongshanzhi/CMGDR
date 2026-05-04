#!/usr/bin/env python3
"""
Phase 4: Aggressive optimization toward MGCN-level performance (1.45x).

Current best: p3_residual_256_lr002 = R@10 0.1794 (1.28x)
LightGCN baseline (LOO): 0.1402
Target: 0.2033 (1.45x)

New strategies:
1. Edge dropout for GCN augmentation (SGL-style)
2. Embedding L2 regularization (standard in LightGCN paper)
3. Higher lr / longer training
4. Multi-negative sampling
5. No text features (text hurts in isolation)
6. No stratified sampling (check impact)
7. Contrastive + best HP combo
8. LR scheduling with warmup
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MODULE_B = str(Path(__file__).resolve().parents[2] / "module_B")
if _MODULE_B not in sys.path:
    sys.path.insert(0, _MODULE_B)

from src.utils import PACKAGE_ROOT, load_config, save_json, ensure_dir, set_seed, resolve_path, active_loss_weights, clone_config


def build_loo_splits(interactions: pd.DataFrame) -> pd.DataFrame:
    interactions = interactions.sort_values(["user_idx", "unixReviewTime"]).copy()
    interactions["split"] = "train"
    for user_idx, group in interactions.groupby("user_idx"):
        idx_list = group.index.tolist()
        if len(idx_list) >= 2:
            interactions.loc[idx_list[-1], "split"] = "test"
            interactions.loc[idx_list[-2], "split"] = "valid"
        elif len(idx_list) == 1:
            interactions.loc[idx_list[-1], "split"] = "test"
    counts = interactions["split"].value_counts()
    print(f"LOO split: train={counts.get('train', 0)}, valid={counts.get('valid', 0)}, test={counts.get('test', 0)}")
    return interactions


def run_experiment_with_features(
    config: dict[str, Any],
    mode: str,
    seed: int,
    suffix: str,
    num_epochs: int = 60,
    edge_drop_rate: float = 0.0,
    l2_reg: float = 0.0,
    use_text: bool = True,
    lr_schedule: str = "constant",  # "constant", "cosine", "warmup_cosine"
) -> dict[str, Any]:
    """Run experiment with LOO split and extended features like edge dropout."""
    import torch
    from torch.utils.data import DataLoader
    from src.data import load_data_bundle, _build_eval_split, _build_user_positive_maps, PairwiseTrainDataset
    from src.features import prepare_visual_artifacts
    from src.models.causal_debias import CMGDRModel
    from src.models.lightgcn_backbone import build_normalized_adjacency
    from src.eval_sampled import sampled_evaluate
    from src.train import _compute_objective

    cfg = clone_config(config)
    cfg["num_epochs"] = num_epochs
    set_seed(seed)

    data_root = resolve_path(PACKAGE_ROOT, cfg["data_root"])
    bundle = load_data_bundle(data_root)
    loo_interactions = build_loo_splits(bundle.interactions)
    bundle.interactions = loo_interactions

    train_inter = loo_interactions[loo_interactions["split"] == "train"]
    n_users = bundle.n_users
    src = train_inter["user_idx"].values.astype(np.int64)
    dst = (train_inter["item_idx"].values + n_users).astype(np.int64)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    edge_index_np = np.stack([row, col], axis=0)

    user_pos_all, user_pos_all_arrays = _build_user_positive_maps(loo_interactions)
    bundle.user_pos_all = user_pos_all
    bundle.user_pos_all_arrays = user_pos_all_arrays
    bundle.eval_splits = {
        "valid": _build_eval_split(loo_interactions, "valid"),
        "test": _build_eval_split(loo_interactions, "test"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = torch.from_numpy(edge_index_np)
    norm_adj = build_normalized_adjacency(edge_index=edge_index, n_nodes=bundle.n_nodes, device=device)

    visual_artifacts = prepare_visual_artifacts(cfg, PACKAGE_ROOT, bundle.n_items, seed)
    visual_features = torch.tensor(visual_artifacts.features, dtype=torch.float32, device=device)
    visual_clusters = torch.tensor(visual_artifacts.clusters, dtype=torch.long, device=device)
    cluster_prototypes = torch.tensor(visual_artifacts.prototypes, dtype=torch.float32, device=device)

    # Text features
    text_features_t = None
    text_dim = 0
    if use_text:
        text_path = resolve_path(Path(data_root), cfg.get("text_feature_path", ""))
        if text_path is not None and text_path.exists():
            text_np = np.load(text_path).astype(np.float32)
            text_dim = text_np.shape[1]
            text_features_t = torch.tensor(text_np, dtype=torch.float32, device=device)
            print(f"Loaded text features: {text_np.shape}")

    # Item-item graph
    item_item_adj = None
    use_ii = cfg.get("use_item_item_graph", False)
    ii_path = resolve_path(Path(data_root), cfg.get("item_item_graph_path", ""))
    if use_ii and ii_path is not None and ii_path.exists():
        from scipy import sparse as sp
        ii_df = pd.read_parquet(ii_path)
        rows_ii = np.concatenate([ii_df["item_idx_a"].values, ii_df["item_idx_b"].values])
        cols_ii = np.concatenate([ii_df["item_idx_b"].values, ii_df["item_idx_a"].values])
        vals_ii = np.ones(len(rows_ii), dtype=np.float32)
        ii_coo = sp.coo_matrix((vals_ii, (rows_ii, cols_ii)), shape=(bundle.n_items, bundle.n_items))
        ii_coo.sum_duplicates()
        deg = np.array(ii_coo.sum(axis=1)).flatten().clip(min=1.0)
        deg_inv_sqrt = np.power(deg, -0.5)
        ii_coo = sp.diags(deg_inv_sqrt) @ ii_coo @ sp.diags(deg_inv_sqrt)
        ii_coo = ii_coo.tocoo()
        indices = torch.tensor(np.stack([ii_coo.row, ii_coo.col]), dtype=torch.long, device=device)
        values = torch.tensor(ii_coo.data, dtype=torch.float32, device=device)
        item_item_adj = torch.sparse_coo_tensor(indices, values, (bundle.n_items, bundle.n_items)).coalesce()
        print(f"Loaded item-item graph: {ii_df.shape[0]} edges")

    # Review ratings
    review_ratings_t = None
    if cfg.get("loss_weights", {}).get("text_consistency", 0.0) > 0:
        avg_rating = train_inter.groupby("item_idx")["overall"].mean()
        rating_arr = np.full(bundle.n_items, -1.0, dtype=np.float32)
        for idx, val in avg_rating.items():
            rating_arr[int(idx)] = (float(val) - 1.0) / 4.0
        review_ratings_t = torch.tensor(rating_arr, dtype=torch.float32, device=device)

    model = CMGDRModel(
        n_users=bundle.n_users,
        n_items=bundle.n_items,
        visual_dim=visual_artifacts.features.shape[1],
        num_clusters=int(visual_artifacts.summary["num_clusters"]),
        embedding_dim=int(cfg["embedding_dim"]),
        num_layers=int(cfg["num_layers"]),
        text_dim=text_dim,
        use_item_item_graph=use_ii,
    ).to(device)

    dataset = PairwiseTrainDataset(
        interactions=loo_interactions,
        n_items=bundle.n_items,
        user_pos_all=bundle.user_pos_all,
        negatives_per_pos=int(cfg.get("negatives_per_pos", 1)),
        seed=seed,
        visual_clusters=visual_artifacts.clusters,
        stratified_sampling=bool(cfg.get("stratified_sampling", False)),
    )
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True)

    weight_map = active_loss_weights(cfg, mode=mode)
    base_lr = float(cfg["learning_rate"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    # LR scheduler
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=base_lr * 0.01)
    elif lr_schedule == "warmup_cosine":
        warmup_epochs = min(5, num_epochs // 10)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    best_valid = float("-inf")
    best_state = None
    print(f"\n{'='*60}")
    print(f"Training: mode={mode}, suffix={suffix}, epochs={num_epochs}")
    print(f"  emb_dim={cfg['embedding_dim']}, lr={base_lr}, edge_drop={edge_drop_rate}, l2_reg={l2_reg}")
    print(f"  use_text={use_text}, use_ii={use_ii}, lr_schedule={lr_schedule}")
    print(f"  weights={weight_map}")
    print(f"  batch_size={cfg['batch_size']}, neg_per_pos={cfg.get('negatives_per_pos', 1)}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in loader:
            batch_user = batch["user_idx"].to(device)
            batch_pos = batch["pos_item_idx"].to(device)
            batch_neg = batch["neg_item_idx"].to(device)
            outputs = model(
                user_indices=batch_user,
                pos_item_indices=batch_pos,
                neg_item_indices=batch_neg,
                norm_adj=norm_adj,
                visual_features=visual_features,
                visual_clusters=visual_clusters,
                cluster_prototypes=cluster_prototypes,
                mode=mode,
                grl_lambda=weight_map["adversarial"],
                text_features=text_features_t,
                item_item_adj=item_item_adj,
                edge_drop_rate=edge_drop_rate,
            )
            loss, metrics = _compute_objective(
                outputs=outputs,
                visual_clusters=visual_clusters,
                weight_map=weight_map,
                review_ratings=review_ratings_t,
                pos_item_indices=batch_pos,
            )
            # L2 regularization on embeddings
            if l2_reg > 0:
                l2_loss = l2_reg * (
                    model.backbone.user_embedding.weight[batch_user].pow(2).mean()
                    + model.backbone.item_embedding.weight[batch_pos].pow(2).mean()
                    + model.backbone.item_embedding.weight[batch_neg].pow(2).mean()
                )
                loss = loss + l2_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(metrics)

        if scheduler is not None:
            scheduler.step()

        avg_loss = sum(m["loss_total"] for m in epoch_losses) / len(epoch_losses)

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            valid_result = sampled_evaluate(
                model=model,
                norm_adj=norm_adj,
                visual_features=visual_features,
                visual_clusters=visual_clusters,
                cluster_prototypes=cluster_prototypes,
                bundle=bundle,
                device=device,
                mode=mode,
                topk_list=[10, 20],
                seed=seed,
                text_features=text_features_t,
                item_item_adj=item_item_adj,
                split_name="valid",
            )
            vr10 = valid_result.get("Recall@10", 0.0)
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | valid R@10={vr10:.4f} | lr={lr_now:.6f}")
            if vr10 >= best_valid:
                best_valid = vr10
                best_state = copy.deepcopy(model.state_dict())
        elif epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f}")

    # Load best model and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_result = sampled_evaluate(
        model=model,
        norm_adj=norm_adj,
        visual_features=visual_features,
        visual_clusters=visual_clusters,
        cluster_prototypes=cluster_prototypes,
        bundle=bundle,
        device=device,
        mode=mode,
        topk_list=[10, 20],
        seed=seed,
        text_features=text_features_t,
        item_item_adj=item_item_adj,
        split_name="test",
    )
    test_result["suffix"] = suffix
    test_result["use_loo"] = True
    test_result["best_valid_R@10"] = best_valid
    return test_result


def main():
    config = load_config(PACKAGE_ROOT / "config" / "model.yaml")
    seed = 42
    results = []
    result_dir = ensure_dir(PACKAGE_ROOT / "results")
    baseline_r10 = 0.1402  # LOO LightGCN baseline

    experiments = [
        # ====== Batch 1: Hyperparameter tuning on best config ======

        # P4-1: Best P3 config (residual 256 lr=0.002) but longer 80ep
        {
            "name": "p4_res256_lr002_80ep",
            "mode": "residual",
            "num_epochs": 80,
            "edge_drop_rate": 0.0,
            "l2_reg": 0.0,
            "use_text": True,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.002,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P4-2: Edge dropout 0.1 (SGL-style augmentation)
        {
            "name": "p4_res256_edgedrop01",
            "mode": "residual",
            "num_epochs": 60,
            "edge_drop_rate": 0.1,
            "l2_reg": 0.0,
            "use_text": True,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.002,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P4-3: Edge dropout 0.2
        {
            "name": "p4_res256_edgedrop02",
            "mode": "residual",
            "num_epochs": 60,
            "edge_drop_rate": 0.2,
            "l2_reg": 0.0,
            "use_text": True,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.002,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P4-4: L2 regularization on embeddings
        {
            "name": "p4_res256_l2reg",
            "mode": "residual",
            "num_epochs": 60,
            "edge_drop_rate": 0.0,
            "l2_reg": 1e-4,
            "use_text": True,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.002,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P4-5: Higher lr=0.003
        {
            "name": "p4_res256_lr003",
            "mode": "residual",
            "num_epochs": 40,
            "edge_drop_rate": 0.0,
            "l2_reg": 0.0,
            "use_text": True,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.003,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P4-6: Cosine LR schedule
        {
            "name": "p4_res256_cosine",
            "mode": "residual",
            "num_epochs": 60,
            "edge_drop_rate": 0.0,
            "l2_reg": 0.0,
            "use_text": True,
            "lr_schedule": "cosine",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.003,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P4-7: No text features (text may hurt)
        {
            "name": "p4_res256_notext",
            "mode": "residual",
            "num_epochs": 40,
            "edge_drop_rate": 0.0,
            "l2_reg": 0.0,
            "use_text": False,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.002,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P4-8: Contrastive + lr=0.002 + edge_drop
        {
            "name": "p4_res256_contr_edgedrop",
            "mode": "residual",
            "num_epochs": 60,
            "edge_drop_rate": 0.1,
            "l2_reg": 0.0,
            "use_text": True,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.002,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0.1,
                    "contrastive_temperature": 0.1, "text_consistency": 0,
                },
            },
        },
        # P4-9: Larger batch 4096 + multi-negative
        {
            "name": "p4_res256_bs4096",
            "mode": "residual",
            "num_epochs": 60,
            "edge_drop_rate": 0.0,
            "l2_reg": 0.0,
            "use_text": True,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.002,
                "num_layers": 3,
                "batch_size": 4096,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P4-10: No residual loss, no orthogonality (pure LightGCN + ii + text, best HP)
        {
            "name": "p4_lightgcn_ii_256_lr002",
            "mode": "lightgcn",
            "num_epochs": 40,
            "edge_drop_rate": 0.0,
            "l2_reg": 0.0,
            "use_text": True,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.002,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # P4-11: Warmup cosine schedule + edge drop + contrastive (kitchen sink)
        {
            "name": "p4_res256_warmup_full",
            "mode": "residual",
            "num_epochs": 80,
            "edge_drop_rate": 0.1,
            "l2_reg": 1e-5,
            "use_text": True,
            "lr_schedule": "warmup_cosine",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.003,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.05, "contrastive": 0.1,
                    "contrastive_temperature": 0.1, "text_consistency": 0,
                },
            },
        },
        # P4-12: No orthogonality (reduce over-regularization)
        {
            "name": "p4_res256_no_orth",
            "mode": "residual",
            "num_epochs": 40,
            "edge_drop_rate": 0.0,
            "l2_reg": 0.0,
            "use_text": True,
            "lr_schedule": "constant",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "learning_rate": 0.002,
                "num_layers": 3,
                "batch_size": 2048,
                "negatives_per_pos": 1,
                "stratified_sampling": False,
                "loss_weights": {
                    "residual": 0.5, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0.0, "contrastive": 0,
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
            result = run_experiment_with_features(
                config=cfg,
                mode=exp["mode"],
                seed=seed,
                suffix=exp["name"],
                num_epochs=exp["num_epochs"],
                edge_drop_rate=exp.get("edge_drop_rate", 0.0),
                l2_reg=exp.get("l2_reg", 0.0),
                use_text=exp.get("use_text", True),
                lr_schedule=exp.get("lr_schedule", "constant"),
            )
            elapsed = time.time() - t0
            result["experiment"] = exp["name"]
            result["elapsed_sec"] = round(elapsed, 1)
            result["config_snapshot"] = {
                "mode": exp["mode"],
                "embedding_dim": cfg["embedding_dim"],
                "learning_rate": cfg["learning_rate"],
                "num_epochs": exp["num_epochs"],
                "num_layers": cfg["num_layers"],
                "edge_drop_rate": exp.get("edge_drop_rate", 0.0),
                "l2_reg": exp.get("l2_reg", 0.0),
                "use_text": exp.get("use_text", True),
                "lr_schedule": exp.get("lr_schedule", "constant"),
                "batch_size": cfg["batch_size"],
                "loss_weights": cfg["loss_weights"],
            }
            results.append(result)

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

        save_json(result_dir / "phase4_results.json", results)

    # Summary
    print("\n" + "=" * 120)
    print("PHASE 4 RESULTS (LOO, Sampled 1+999)")
    print("=" * 120)
    print(f"LightGCN baseline: {baseline_r10:.4f}")
    print(f"Phase 3 best:      0.1794 (1.28x)")
    print(f"MGCN target:       {baseline_r10*1.45:.4f} (1.45x)")
    print()
    print(f"{'Experiment':<35} {'R@10':>8} {'R@20':>8} {'N@10':>8} {'N@20':>8} {'ratio':>8} {'time':>6}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: x.get('Recall@10', 0), reverse=True):
        if "error" not in r:
            r10 = r.get('Recall@10', 0)
            print(f"  {r['experiment']:<35} {r10:>7.4f} {r.get('Recall@20',0):>8.4f} "
                  f"{r.get('NDCG@10',0):>8.4f} {r.get('NDCG@20',0):>8.4f} "
                  f"{r10/baseline_r10:>7.2f}x {r.get('elapsed_sec',0):>5.0f}s")
    print("=" * 120)

    save_json(result_dir / "phase4_results.json", results)


if __name__ == "__main__":
    main()
