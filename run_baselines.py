#!/usr/bin/env python3
"""
Run multimodal recommendation baselines (MMGCN, LATTICE, BM3, FREEDOM, MGCN)
with the same LOO protocol and sampled evaluation as CMGDR.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "module_B"))

from src.utils import PACKAGE_ROOT, load_config, set_seed, resolve_path, save_json, ensure_dir, clone_config
from src.losses import bpr_loss
from src.eval_sampled import sampled_evaluate
from src.data import load_data_bundle, _build_eval_split, _build_user_positive_maps, PairwiseTrainDataset
from src.features import prepare_visual_artifacts
from src.metrics import cluster_exposure_gap, cluster_calibration_gap, cluster_probe_accuracy
from src.models.lightgcn_backbone import build_normalized_adjacency
from src.models.baselines import MMGCNModel, LATTICEModel, BM3Model, FREEDOMModel, MGCNModel


def build_loo_splits(interactions):
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


class BaselineWrapper:
    """Wraps baseline models to match CMGDR's evaluation interface."""

    def __init__(self, model, norm_adj, visual_features, text_features):
        self.model = model
        self.norm_adj = norm_adj
        self.visual_features = visual_features
        self.text_features = text_features

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def encode_all(self, **kwargs):
        user_emb, item_emb = self.model.encode_all(
            self.norm_adj, self.visual_features, self.text_features
        )
        return {
            "user_embeddings": user_emb,
            "item_total_embeddings": item_emb,
            "item_causal_embeddings": item_emb,
            "item_graph_embeddings": item_emb,
            "item_bias_embeddings": torch.zeros_like(item_emb),
            "counterfactual_causal_embeddings": item_emb,
            "counterfactual_total_embeddings": item_emb,
            "mode": "lightgcn",
        }


def run_baseline(model_class, model_name, config, seed, num_epochs=40, extra_kwargs=None):
    set_seed(seed)
    cfg = clone_config(config)
    data_root = resolve_path(PACKAGE_ROOT, cfg["data_root"])
    bundle = load_data_bundle(data_root)

    loo_interactions = build_loo_splits(bundle.interactions)
    bundle.interactions = loo_interactions
    train_inter = loo_interactions[loo_interactions["split"] == "train"]
    n_users = bundle.n_users

    src_arr = train_inter["user_idx"].values.astype(np.int64)
    dst_arr = (train_inter["item_idx"].values + n_users).astype(np.int64)
    row = np.concatenate([src_arr, dst_arr])
    col = np.concatenate([dst_arr, src_arr])
    edge_index = torch.from_numpy(np.stack([row, col], axis=0))

    user_pos_all, user_pos_all_arrays = _build_user_positive_maps(loo_interactions)
    bundle.user_pos_all = user_pos_all
    bundle.user_pos_all_arrays = user_pos_all_arrays
    bundle.eval_splits = {
        "valid": _build_eval_split(loo_interactions, "valid"),
        "test": _build_eval_split(loo_interactions, "test"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_adj = build_normalized_adjacency(edge_index, bundle.n_nodes, device)

    visual_artifacts = prepare_visual_artifacts(cfg, PACKAGE_ROOT, bundle.n_items, seed)
    visual_features = torch.tensor(visual_artifacts.features, dtype=torch.float32, device=device)
    visual_clusters = torch.tensor(visual_artifacts.clusters, dtype=torch.long, device=device)

    text_path = resolve_path(PACKAGE_ROOT, cfg.get("text_feature_path", ""))
    text_np = np.load(text_path).astype(np.float32)
    text_features = torch.tensor(text_np, dtype=torch.float32, device=device)
    print(f"Loaded: visual={visual_features.shape}, text={text_features.shape}")

    # Instantiate model
    kwargs = dict(
        n_users=bundle.n_users, n_items=bundle.n_items,
        visual_dim=visual_features.shape[1], text_dim=text_features.shape[1],
        embedding_dim=256, num_layers=3,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    model = model_class(**kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name}: {n_params:,} parameters")

    dataset = PairwiseTrainDataset(
        interactions=loo_interactions, n_items=bundle.n_items,
        user_pos_all=bundle.user_pos_all, negatives_per_pos=1, seed=seed,
    )
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    wrapper = BaselineWrapper(model, norm_adj, visual_features, text_features)

    best_valid = float("-inf")
    best_state = None

    print(f"\n{'='*70}")
    print(f"Training: {model_name}, epochs={num_epochs}, emb=256, lr=0.001")
    print(f"{'='*70}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        for batch in loader:
            u = batch["user_idx"].to(device)
            pos = batch["pos_item_idx"].to(device)
            neg = batch["neg_item_idx"].to(device)

            user_emb, item_emb = model.encode_all(norm_adj, visual_features, text_features)

            pos_scores = (user_emb[u] * item_emb[pos]).sum(dim=-1)
            neg_scores = (user_emb[u] * item_emb[neg]).sum(dim=-1)
            loss = bpr_loss(pos_scores, neg_scores)

            # BM3 extra contrastive loss
            if isinstance(model, BM3Model):
                loss = loss + model.contrastive_loss(visual_features, text_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1

        if epoch % 5 == 0 or epoch == num_epochs:
            valid_result = sampled_evaluate(
                model=wrapper, norm_adj=norm_adj,
                visual_features=visual_features, visual_clusters=visual_clusters,
                cluster_prototypes=torch.tensor(visual_artifacts.prototypes, dtype=torch.float32, device=device),
                bundle=bundle, device=device, mode="lightgcn",
                topk_list=[10, 20], seed=seed,
                text_features=text_features, item_item_adj=None,
                split_name="valid",
            )
            vr10 = valid_result.get("Recall@10", 0.0)
            print(f"  Epoch {epoch:3d} | loss={epoch_loss/n_batch:.4f} | valid R@10={vr10:.4f}")
            if vr10 >= best_valid:
                best_valid = vr10
                best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_dir = ensure_dir(Path(__file__).resolve().parent / "shared_data" / "model_outputs" / "checkpoints")
    ckpt_path = ckpt_dir / f"{model_name}_seed{seed}.pt"
    torch.save({
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "best_valid_R@10": best_valid,
        "n_params": n_params,
        "embedding_dim": kwargs.get("embedding_dim", 256),
    }, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    test_result = sampled_evaluate(
        model=wrapper, norm_adj=norm_adj,
        visual_features=visual_features, visual_clusters=visual_clusters,
        cluster_prototypes=torch.tensor(visual_artifacts.prototypes, dtype=torch.float32, device=device),
        bundle=bundle, device=device, mode="lightgcn",
        topk_list=[10, 20], seed=seed,
        text_features=text_features, item_item_adj=None,
        split_name="test",
    )
    test_result["model"] = model_name
    test_result["best_valid_R@10"] = best_valid
    test_result["n_params"] = n_params

    # === Debiasing metrics (same as CMGDR evaluation) ===
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.encode_all(norm_adj, visual_features, text_features)

    vc_np = visual_clusters.cpu().numpy()
    test_split = bundle.eval_splits["test"]
    recs_by_user = {}
    targets_by_user_dict = {}

    for user_idx in test_split.targets_by_user:
        u_emb = user_emb[user_idx].unsqueeze(0)
        scores = (u_emb @ item_emb.T).squeeze(0)
        if user_idx in bundle.user_pos_all:
            for pos_item in bundle.user_pos_all[user_idx]:
                scores[pos_item] = -1e9
        _, topk_idx = torch.topk(scores, 20)
        recs_by_user[user_idx] = topk_idx.cpu().numpy()
        targets_by_user_dict[user_idx] = test_split.targets_by_user[user_idx]

    exp_gap, _ = cluster_exposure_gap(recs_by_user, vc_np, topk=10)
    cal_gap, _ = cluster_calibration_gap(recs_by_user, targets_by_user_dict, vc_np, topk=10)
    probe_acc = cluster_probe_accuracy(item_emb.cpu().numpy(), vc_np)

    test_result["exposure_gap@10"] = exp_gap
    test_result["calibration_gap@10"] = cal_gap
    test_result["probe_item_emb"] = probe_acc

    print(f"\n>>> {model_name}: R@10={test_result['Recall@10']:.4f}, R@20={test_result['Recall@20']:.4f}, "
          f"N@10={test_result['NDCG@10']:.4f}, N@20={test_result['NDCG@20']:.4f}")
    print(f"    ExpGap={exp_gap:.4f}, CalGap={cal_gap:.4f}, Probe={probe_acc:.3f}")
    return test_result


def main():
    config = load_config(PACKAGE_ROOT / "config" / "model.yaml")
    seed = 42
    results = []
    t_total = time.time()

    baselines = [
        (MMGCNModel, "MMGCN", {}),
        (LATTICEModel, "LATTICE", {"k_neighbors": 10}),
        (BM3Model, "BM3", {"cl_weight": 0.1, "temperature": 0.2, "dropout": 0.3}),
        (FREEDOMModel, "FREEDOM", {"k_neighbors": 10}),
        (MGCNModel, "MGCN", {}),
    ]

    for model_class, name, extra_kwargs in baselines:
        t0 = time.time()
        result = run_baseline(model_class, name, config, seed, num_epochs=40, extra_kwargs=extra_kwargs)
        result["time_sec"] = time.time() - t0
        results.append(result)

    # Print summary
    print(f"\n{'='*100}")
    print(f"BASELINE RESULTS — Sports & Outdoors (LOO, Sampled 1+999)")
    print(f"{'='*100}")
    print(f"{'Method':<12} {'R@10':>7} {'R@20':>7} {'N@10':>7} {'N@20':>7} {'HR@10':>7} {'MRR':>7} "
          f"{'ExpGap':>7} {'CalGap':>7} {'Probe':>7} {'Params':>10}")
    print("-" * 105)
    for r in results:
        print(f"{r['model']:<12} {r['Recall@10']:7.4f} {r['Recall@20']:7.4f} "
              f"{r['NDCG@10']:7.4f} {r['NDCG@20']:7.4f} {r['HR@10']:7.4f} {r['MRR']:7.4f} "
              f"{r.get('exposure_gap@10',0):7.4f} {r.get('calibration_gap@10',0):7.4f} "
              f"{r.get('probe_item_emb',0):7.3f} {r['n_params']:>10,}")
    print(f"{'='*100}")
    print(f"Total time: {time.time() - t_total:.0f}s")

    # Add our CMGDR results for comparison
    cmgdr_path = Path(__file__).resolve().parent / "shared_data" / "evaluation" / "full_evaluation_results.json"
    if cmgdr_path.exists():
        cmgdr_results = json.loads(cmgdr_path.read_text())
        print(f"\n--- Combined with CMGDR ---")
        all_results = results + cmgdr_results
        lgcn_r10 = next((r["Recall@10"] for r in cmgdr_results if r["suffix"] == "LightGCN"), 0.1391)
        print(f"{'Method':<20} {'R@10':>7} {'R@20':>7} {'N@10':>7} {'N@20':>7} {'vs LightGCN':>12}")
        print("-" * 70)
        for r in sorted(all_results, key=lambda x: x.get("Recall@10", 0)):
            name = r.get("model", r.get("suffix", "?"))
            ratio = r["Recall@10"] / lgcn_r10 if lgcn_r10 > 0 else 0
            print(f"{name:<20} {r['Recall@10']:7.4f} {r['Recall@20']:7.4f} "
                  f"{r['NDCG@10']:7.4f} {r['NDCG@20']:7.4f} {ratio:11.2f}x")

    out_dir = ensure_dir(Path(__file__).resolve().parent / "shared_data" / "evaluation")
    save_json(out_dir / "baseline_results.json", results)
    print(f"\nResults saved to: {out_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
