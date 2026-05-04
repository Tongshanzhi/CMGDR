#!/usr/bin/env python3
"""
CMGDR Full Evaluation: LOO protocol + debiasing metrics on Sports & Outdoors.

Runs all ablation modes required by the proposal:
1. LightGCN baseline (ID-only)
2. LightGCN + text + item-item graph (multimodal, no debias)
3. CMGDR-visual (visual debias only)
4. CMGDR-multimodal (visual + text + ii-graph debias)
5. CMGDR-full (all components, best config from optimization)

Evaluation protocol: Leave-One-Out, sampled 1+999 (SOTA-comparable).
Metrics: Recall@K, NDCG@K, HR@K, MRR + debiasing metrics.
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "module_B"))

from src.utils import PACKAGE_ROOT, load_config, set_seed, resolve_path, save_json, ensure_dir, clone_config, active_loss_weights


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


def run_experiment(config, mode, seed, suffix, num_epochs, overrides=None):
    """Run single experiment with LOO + sampled eval + debiasing metrics."""
    import torch
    from torch.utils.data import DataLoader
    from src.train import _compute_objective
    from src.eval_sampled import sampled_evaluate
    from src.data import load_data_bundle, _build_eval_split, _build_user_positive_maps, PairwiseTrainDataset, build_symmetric_edge_index
    from src.features import prepare_visual_artifacts
    from src.models.causal_debias import CMGDRModel
    from src.models.lightgcn_backbone import build_normalized_adjacency
    from src.metrics import cluster_exposure_gap, cluster_calibration_gap, cluster_probe_accuracy

    cfg = clone_config(config)
    if overrides:
        cfg.update(overrides)
    cfg["num_epochs"] = num_epochs

    set_seed(seed)
    data_root = resolve_path(PACKAGE_ROOT, cfg["data_root"])
    bundle = load_data_bundle(data_root)

    # LOO split
    loo_interactions = build_loo_splits(bundle.interactions)
    bundle.interactions = loo_interactions
    train_inter = loo_interactions[loo_interactions["split"] == "train"]
    n_users = bundle.n_users

    # Rebuild edges from LOO train
    src_arr = train_inter["user_idx"].values.astype(np.int64)
    dst_arr = (train_inter["item_idx"].values + n_users).astype(np.int64)
    row = np.concatenate([src_arr, dst_arr])
    col = np.concatenate([dst_arr, src_arr])
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
        n_users=bundle.n_users, n_items=bundle.n_items,
        visual_dim=visual_artifacts.features.shape[1],
        num_clusters=int(visual_artifacts.summary["num_clusters"]),
        embedding_dim=int(cfg["embedding_dim"]),
        num_layers=int(cfg["num_layers"]),
        text_dim=text_dim, use_item_item_graph=use_ii,
    ).to(device)

    dataset = PairwiseTrainDataset(
        interactions=loo_interactions, n_items=bundle.n_items,
        user_pos_all=bundle.user_pos_all,
        negatives_per_pos=int(cfg.get("negatives_per_pos", 1)),
        seed=seed, visual_clusters=visual_artifacts.clusters,
        stratified_sampling=bool(cfg.get("stratified_sampling", False)),
    )
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True)
    weight_map = active_loss_weights(cfg, mode=mode)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )

    # Training
    best_valid = float("-inf")
    best_state = None
    print(f"\n{'='*70}")
    print(f"Training: mode={mode}, suffix={suffix}, epochs={num_epochs}")
    print(f"  emb_dim={cfg['embedding_dim']}, lr={cfg['learning_rate']}, weights={weight_map}")
    print(f"{'='*70}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in loader:
            batch_user = batch["user_idx"].to(device)
            batch_pos = batch["pos_item_idx"].to(device)
            batch_neg = batch["neg_item_idx"].to(device)
            outputs = model(
                user_indices=batch_user, pos_item_indices=batch_pos,
                neg_item_indices=batch_neg, norm_adj=norm_adj,
                visual_features=visual_features, visual_clusters=visual_clusters,
                cluster_prototypes=cluster_prototypes, mode=mode,
                grl_lambda=weight_map["adversarial"],
                text_features=text_features_t, item_item_adj=item_item_adj,
            )
            loss, metrics = _compute_objective(
                outputs=outputs, visual_clusters=visual_clusters,
                weight_map=weight_map, review_ratings=review_ratings_t,
                pos_item_indices=batch_pos,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(metrics)

        avg_loss = sum(m["loss_total"] for m in epoch_losses) / len(epoch_losses)

        if epoch % 5 == 0 or epoch == num_epochs:
            valid_result = sampled_evaluate(
                model=model, norm_adj=norm_adj,
                visual_features=visual_features, visual_clusters=visual_clusters,
                cluster_prototypes=cluster_prototypes, bundle=bundle,
                device=device, mode=mode, topk_list=[10, 20], seed=seed,
                text_features=text_features_t, item_item_adj=item_item_adj,
                split_name="valid",
            )
            vr10 = valid_result.get("Recall@10", 0.0)
            print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | valid R@10={vr10:.4f}")
            if vr10 >= best_valid:
                best_valid = vr10
                best_state = copy.deepcopy(model.state_dict())

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_dir = ensure_dir(Path(__file__).resolve().parent / "shared_data" / "model_outputs" / "checkpoints")
    import torch as _torch
    ckpt_path = ckpt_dir / f"{suffix}_seed{seed}.pt"
    _torch.save({
        "model_name": suffix,
        "mode": mode,
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "best_valid_R@10": best_valid,
        "embedding_dim": int(cfg["embedding_dim"]),
        "config": cfg,
    }, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Test evaluation (sampled)
    test_result = sampled_evaluate(
        model=model, norm_adj=norm_adj,
        visual_features=visual_features, visual_clusters=visual_clusters,
        cluster_prototypes=cluster_prototypes, bundle=bundle,
        device=device, mode=mode, topk_list=[10, 20], seed=seed,
        text_features=text_features_t, item_item_adj=item_item_adj,
        split_name="test",
    )

    # === Debiasing metrics ===
    model.eval()
    with torch.no_grad():
        enc = model.encode_all(
            norm_adj=norm_adj, visual_features=visual_features,
            visual_clusters=visual_clusters, cluster_prototypes=cluster_prototypes,
            mode=mode, text_features=text_features_t, item_item_adj=item_item_adj,
        )
    user_emb = enc["user_embeddings"]
    item_total = enc["item_total_embeddings"]
    item_causal = enc["item_causal_embeddings"]
    item_graph = enc["item_graph_embeddings"]
    cf_total = enc["counterfactual_total_embeddings"]

    # Build recommendations for test users (top-20)
    test_split = bundle.eval_splits["test"]
    recs_by_user = {}
    targets_by_user = {}
    vc_np = visual_clusters.cpu().numpy()

    for user_idx in test_split.targets_by_user:
        u_emb = user_emb[user_idx].unsqueeze(0)
        scores = (u_emb @ item_total.T).squeeze(0)
        # Mask training positives
        if user_idx in bundle.user_pos_all:
            for pos_item in bundle.user_pos_all[user_idx]:
                scores[pos_item] = -1e9
        _, topk_idx = torch.topk(scores, 20)
        recs_by_user[user_idx] = topk_idx.cpu().numpy()
        targets_by_user[user_idx] = test_split.targets_by_user[user_idx]

    # Exposure gap & calibration gap
    exp_gap, _ = cluster_exposure_gap(recs_by_user, vc_np, topk=10)
    cal_gap, _ = cluster_calibration_gap(recs_by_user, targets_by_user, vc_np, topk=10)

    # Probe accuracy (can causal embeddings predict visual cluster?)
    probe_graph = cluster_probe_accuracy(item_graph.cpu().numpy(), vc_np)
    probe_causal = cluster_probe_accuracy(item_causal.cpu().numpy(), vc_np)

    # Counterfactual score shift
    causal_scores = (user_emb @ item_causal.T)
    cf_scores = (user_emb @ enc["counterfactual_causal_embeddings"].T)
    cf_shift = (causal_scores - cf_scores).abs().mean().item()

    test_result.update({
        "suffix": suffix,
        "mode": mode,
        "best_valid_R@10": best_valid,
        "exposure_gap@10": exp_gap,
        "calibration_gap@10": cal_gap,
        "probe_graph": probe_graph,
        "probe_causal": probe_causal,
        "cf_score_shift": cf_shift,
    })

    print(f"\n>>> {suffix}: R@10={test_result['Recall@10']:.4f}, R@20={test_result['Recall@20']:.4f}, "
          f"N@10={test_result['NDCG@10']:.4f}, N@20={test_result['NDCG@20']:.4f}")
    print(f"    ExpGap={exp_gap:.4f}, CalGap={cal_gap:.4f}, "
          f"Probe_g={probe_graph:.3f}, Probe_c={probe_causal:.3f}, CF_shift={cf_shift:.4f}")

    return test_result


def main():
    config = load_config(PACKAGE_ROOT / "config" / "model.yaml")
    seed = 42
    results = []
    t_total = time.time()

    # ========== Experiment Suite ==========

    experiments = [
        # 1. LightGCN baseline (ID-only, no visual, no text, no ii-graph)
        {
            "name": "LightGCN",
            "mode": "lightgcn",
            "epochs": 40,
            "overrides": {
                "use_item_item_graph": False,
                "embedding_dim": 128,
                "learning_rate": 0.001,
                "loss_weights": {
                    "residual": 0, "adversarial": 0, "counterfactual": 0,
                    "orthogonality": 0, "contrastive": 0,
                    "contrastive_temperature": 0.2, "text_consistency": 0,
                },
            },
        },
        # 2. Multimodal baseline (text + ii-graph, no debias)
        {
            "name": "MM-LightGCN",
            "mode": "lightgcn",
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
        # 3. Visual-Concat (naive multimodal fusion, exposes visual bias)
        {
            "name": "Visual-Concat",
            "mode": "visual_concat",
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
        # 4. CMGDR-Residual (causal decomposition only)
        {
            "name": "CMGDR-Residual",
            "mode": "residual",
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
        # 5. CMGDR-Full (all components: residual + adversarial + counterfactual + orthogonality)
        {
            "name": "CMGDR-Full",
            "mode": "full",
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
    ]

    for exp in experiments:
        t0 = time.time()
        result = run_experiment(
            config=config, mode=exp["mode"], seed=seed,
            suffix=exp["name"], num_epochs=exp["epochs"],
            overrides=exp.get("overrides"),
        )
        result["time_sec"] = time.time() - t0
        results.append(result)

    # ========== Print Summary ==========
    baseline_r10 = results[0]["Recall@10"]

    print(f"\n{'='*100}")
    print(f"CMGDR EVALUATION RESULTS — Sports & Outdoors (LOO, Sampled 1+999)")
    print(f"{'='*100}")
    print(f"{'Method':<20} {'R@10':>7} {'R@20':>7} {'N@10':>7} {'N@20':>7} {'HR@10':>7} {'MRR':>7} "
          f"{'vs Base':>8} {'ExpGap':>7} {'CalGap':>7} {'Probe_c':>8} {'CF_Shift':>9}")
    print("-" * 120)
    for r in results:
        ratio = r["Recall@10"] / baseline_r10 if baseline_r10 > 0 else 0
        print(f"{r['suffix']:<20} {r['Recall@10']:7.4f} {r['Recall@20']:7.4f} "
              f"{r['NDCG@10']:7.4f} {r['NDCG@20']:7.4f} {r['HR@10']:7.4f} {r['MRR']:7.4f} "
              f"{ratio:7.2f}x {r['exposure_gap@10']:7.4f} {r['calibration_gap@10']:7.4f} "
              f"{r['probe_causal']:8.3f} {r['cf_score_shift']:9.4f}")
    print(f"{'='*100}")
    print(f"Total time: {time.time() - t_total:.0f}s")

    # Save results
    out_dir = ensure_dir(Path(__file__).resolve().parent / "shared_data" / "evaluation")
    save_json(out_dir / "full_evaluation_results.json", results)
    print(f"Results saved to: {out_dir / 'full_evaluation_results.json'}")


if __name__ == "__main__":
    main()
