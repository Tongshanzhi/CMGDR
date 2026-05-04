#!/usr/bin/env python3
"""
Comprehensive experiment runner for CMGDR.
Runs ablation experiments with Leave-One-Out evaluation (SOTA-comparable).
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

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import PACKAGE_ROOT, load_config, set_seed, resolve_path, save_json, ensure_dir


def build_loo_splits(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-One-Out split: for each user, the chronologically last interaction
    goes to test, second-to-last goes to valid, rest to train.
    This matches the protocol used by MMGCN, LATTICE, BM3, FREEDOM, MGCN.
    """
    interactions = interactions.sort_values(["user_idx", "unixReviewTime"]).copy()
    interactions["split"] = "train"

    # For each user: last -> test, second-to-last -> valid
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


def run_single_experiment(
    config: dict[str, Any],
    mode: str,
    seed: int,
    suffix: str,
    use_loo: bool = False,
    num_epochs: int | None = None,
) -> dict[str, Any]:
    """Run a single training + sampled evaluation experiment."""
    from src.train import run_training, _prepare_runtime, _compute_objective
    from src.eval_sampled import sampled_evaluate
    from src.utils import active_loss_weights, clone_config
    import torch
    from torch.utils.data import DataLoader

    cfg = clone_config(config)
    if num_epochs is not None:
        cfg["num_epochs"] = num_epochs

    set_seed(seed)

    # If LOO, re-split the data
    if use_loo:
        from src.data import load_data_bundle, _build_eval_split, _build_user_positive_maps, PairwiseTrainDataset
        from src.features import prepare_visual_artifacts
        from src.models.causal_debias import CMGDRModel
        from src.models.lightgcn_backbone import build_normalized_adjacency
        from src.data import build_symmetric_edge_index

        data_root = resolve_path(PACKAGE_ROOT, cfg["data_root"])
        bundle = load_data_bundle(data_root)

        # Re-split interactions using LOO
        loo_interactions = build_loo_splits(bundle.interactions)
        bundle.interactions = loo_interactions

        # Rebuild edge_train from LOO train split
        train_inter = loo_interactions[loo_interactions["split"] == "train"]
        n_users = bundle.n_users
        edges_train_df = pd.DataFrame({
            "src": train_inter["user_idx"].values + 0,  # user global ID = user_idx
            "dst": train_inter["item_idx"].values + n_users,  # item global ID = n_users + item_idx
        })
        # Build symmetric edge index
        src = edges_train_df["src"].to_numpy(dtype=np.int64)
        dst = edges_train_df["dst"].to_numpy(dtype=np.int64)
        row = np.concatenate([src, dst])
        col = np.concatenate([dst, src])
        edge_index_np = np.stack([row, col], axis=0)

        # Rebuild user_pos_all
        user_pos_all, user_pos_all_arrays = _build_user_positive_maps(loo_interactions)
        bundle.user_pos_all = user_pos_all
        bundle.user_pos_all_arrays = user_pos_all_arrays

        # Rebuild eval splits
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

        # Load text features
        text_features_t = None
        text_dim = 0
        text_path = resolve_path(Path(data_root), cfg.get("text_feature_path", ""))
        if text_path is not None and text_path.exists():
            text_np = np.load(text_path).astype(np.float32)
            text_dim = text_np.shape[1]
            text_features_t = torch.tensor(text_np, dtype=torch.float32, device=device)
            print(f"Loaded text features: {text_np.shape}")

        # Load item-item graph
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

        # Build review ratings
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
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )

        # Training
        best_valid = float("-inf")
        best_state = None
        n_epochs = int(cfg["num_epochs"])
        print(f"\n{'='*60}")
        print(f"Training: mode={mode}, suffix={suffix}, epochs={n_epochs}, LOO={use_loo}")
        print(f"  emb_dim={cfg['embedding_dim']}, lr={cfg['learning_rate']}, weights={weight_map}")
        print(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
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
                )
                loss, metrics = _compute_objective(
                    outputs=outputs,
                    visual_clusters=visual_clusters,
                    weight_map=weight_map,
                    review_ratings=review_ratings_t,
                    pos_item_indices=batch_pos,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(metrics)

            avg_loss = sum(m["loss_total"] for m in epoch_losses) / len(epoch_losses)

            # Validate every 5 epochs
            if epoch % 5 == 0 or epoch == n_epochs:
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
                print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | valid R@10={vr10:.4f}")
                if vr10 >= best_valid:
                    best_valid = vr10
                    best_state = copy.deepcopy(model.state_dict())
            else:
                if epoch % 10 == 0:
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
        test_result["use_loo"] = use_loo
        test_result["best_valid_R@10"] = best_valid
        return test_result

    else:
        # Use temporal split (existing pipeline)
        summary = run_training(
            config_or_path=cfg,
            seed=seed,
            override_mode=mode,
            suffix=suffix,
            save_outputs=True,
        )
        # Run sampled eval on test set
        runtime = _prepare_runtime(cfg, seed, mode)
        test_result = sampled_evaluate(
            model=runtime["model"],
            norm_adj=runtime["norm_adj"],
            visual_features=runtime["visual_features"],
            visual_clusters=runtime["visual_clusters"],
            cluster_prototypes=runtime["cluster_prototypes"],
            bundle=runtime["bundle"],
            device=runtime["device"],
            mode=mode,
            topk_list=[10, 20],
            seed=seed,
            text_features=runtime.get("text_features"),
            item_item_adj=runtime.get("item_item_adj"),
        )
        test_result["suffix"] = suffix
        test_result["use_loo"] = False
        return test_result


def main():
    config = load_config(PACKAGE_ROOT / "config" / "model.yaml")
    seed = 42
    results = []
    result_dir = ensure_dir(PACKAGE_ROOT / "results")

    # ===== Experiment Group 1: Ablation with LOO (SOTA-comparable) =====
    experiments = [
        # Exp 1: LightGCN baseline (no visual, no text, no ii-graph)
        {
            "name": "loo_lightgcn",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": False,
                "embedding_dim": 128,
                "num_epochs": 40,
                "loss_weights": {"residual": 0, "adversarial": 0, "counterfactual": 0, "orthogonality": 0, "contrastive": 0, "text_consistency": 0},
            },
        },
        # Exp 2: LightGCN + text only
        {
            "name": "loo_text_only",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": False,
                "embedding_dim": 128,
                "num_epochs": 40,
                "loss_weights": {"residual": 0, "adversarial": 0, "counterfactual": 0, "orthogonality": 0, "contrastive": 0, "text_consistency": 0},
            },
        },
        # Exp 3: LightGCN + text + item-item graph
        {
            "name": "loo_text_ii",
            "mode": "lightgcn",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 40,
                "loss_weights": {"residual": 0, "adversarial": 0, "counterfactual": 0, "orthogonality": 0, "contrastive": 0, "text_consistency": 0},
            },
        },
        # Exp 4: CMGDR full (visual only, no text, no ii-graph) - previous best
        {
            "name": "loo_cmgdr_visual",
            "mode": "full",
            "overrides": {
                "use_item_item_graph": False,
                "embedding_dim": 128,
                "num_epochs": 40,
                "loss_weights": {"residual": 1.0, "adversarial": 0.01, "counterfactual": 0.5, "orthogonality": 0.1, "contrastive": 0, "text_consistency": 0},
            },
        },
        # Exp 5: CMGDR + text + ii-graph (all modalities, no contrastive)
        {
            "name": "loo_cmgdr_multimodal",
            "mode": "full",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 40,
                "loss_weights": {"residual": 1.0, "adversarial": 0.01, "counterfactual": 0.5, "orthogonality": 0.1, "contrastive": 0, "text_consistency": 0},
            },
        },
        # Exp 6: CMGDR + contrastive (all components)
        {
            "name": "loo_cmgdr_contrastive",
            "mode": "full",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 40,
                "loss_weights": {"residual": 1.0, "adversarial": 0.01, "counterfactual": 0.5, "orthogonality": 0.1, "contrastive": 0.1, "contrastive_temperature": 0.2, "text_consistency": 0},
            },
        },
        # Exp 7: CMGDR-full all components
        {
            "name": "loo_cmgdr_full",
            "mode": "full",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 128,
                "num_epochs": 40,
                "stratified_sampling": True,
                "loss_weights": {"residual": 1.0, "adversarial": 0.01, "counterfactual": 0.5, "orthogonality": 0.1, "contrastive": 0.1, "contrastive_temperature": 0.2, "text_consistency": 0.05},
            },
        },
        # Exp 8: CMGDR-full with higher capacity (emb=256)
        {
            "name": "loo_cmgdr_full_256",
            "mode": "full",
            "overrides": {
                "use_item_item_graph": True,
                "embedding_dim": 256,
                "num_epochs": 40,
                "stratified_sampling": True,
                "loss_weights": {"residual": 1.0, "adversarial": 0.01, "counterfactual": 0.5, "orthogonality": 0.1, "contrastive": 0.1, "contrastive_temperature": 0.2, "text_consistency": 0.05},
            },
        },
    ]

    for exp in experiments:
        cfg = copy.deepcopy(config)
        for k, v in exp["overrides"].items():
            cfg[k] = v

        t0 = time.time()
        try:
            result = run_single_experiment(
                config=cfg,
                mode=exp["mode"],
                seed=seed,
                suffix=exp["name"],
                use_loo=True,
                num_epochs=cfg.get("num_epochs", 40),
            )
            elapsed = time.time() - t0
            result["experiment"] = exp["name"]
            result["elapsed_sec"] = round(elapsed, 1)
            results.append(result)

            print(f"\n>>> {exp['name']}: R@10={result.get('Recall@10', 0):.4f}, "
                  f"R@20={result.get('Recall@20', 0):.4f}, "
                  f"N@10={result.get('NDCG@10', 0):.4f}, "
                  f"N@20={result.get('NDCG@20', 0):.4f}, "
                  f"time={elapsed:.0f}s\n")
        except Exception as e:
            print(f"\n>>> {exp['name']} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            results.append({"experiment": exp["name"], "error": str(e)})

        # Save intermediate results
        save_json(result_dir / "loo_experiment_results.json", results)

    # ===== Print summary table =====
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY (Leave-One-Out, Sampled 1+999)")
    print("=" * 100)
    header = f"{'Experiment':<30} {'R@10':>8} {'R@20':>8} {'N@10':>8} {'N@20':>8} {'HR@10':>8} {'MRR':>8}"
    print(header)
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['experiment']:<30} FAILED: {r['error']}")
        else:
            print(f"{r['experiment']:<30} "
                  f"{r.get('Recall@10', 0):>8.4f} "
                  f"{r.get('Recall@20', 0):>8.4f} "
                  f"{r.get('NDCG@10', 0):>8.4f} "
                  f"{r.get('NDCG@20', 0):>8.4f} "
                  f"{r.get('HR@10', 0):>8.4f} "
                  f"{r.get('MRR', 0):>8.4f}")
    print("=" * 100)

    # Compare with published SOTA
    print("\n--- Published SOTA (Leave-One-Out, Sports & Outdoors) ---")
    sota = [
        ("LightGCN", 0.0548),
        ("MMGCN", 0.0524),
        ("GRCN", 0.0617),
        ("LATTICE", 0.0695),
        ("BM3", 0.0728),
        ("FREEDOM", 0.0766),
        ("MGCN", 0.0792),
    ]
    for name, r10 in sota:
        print(f"  {name:<20} R@10={r10:.4f}  ({r10/0.0548:.2f}x vs LightGCN)")

    save_json(result_dir / "loo_experiment_results.json", results)
    print(f"\nResults saved to: {result_dir / 'loo_experiment_results.json'}")


if __name__ == "__main__":
    main()
