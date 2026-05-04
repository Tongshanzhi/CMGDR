"""Section 7.8 — Simulated A/B with Inverse-Propensity-Weighted NDCG.

The standard sampled-1+999 metric mixes ranking quality with exposure bias.
A propensity-aware estimator weights user--item pairs by 1/P(V=v) where v
is the visual cluster of the test target — over-represented clusters get
down-weighted, under-represented clusters up-weighted. This yields a
counter-factual estimator of NDCG under a uniform-cluster exposure policy
(see Schnabel et al., "Recommendations as Treatments", ICML'16).

We compute IPW-NDCG@10 for every method whose checkpoint is on disk,
using the visual cluster prior P(V=v) on the catalogue. CMGDR's promise
is that its causal embedding eats less of the visual confounding -> its
IPW-NDCG should be closer to the unweighted NDCG, while methods with
strong visual leakage should drop significantly.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module_B"))

from src.utils import PACKAGE_ROOT, load_config, ensure_dir, save_json, set_seed, resolve_path
from src.data import load_data_bundle, _build_eval_split, _build_user_positive_maps
from src.features import prepare_visual_artifacts
from src.models.causal_debias import CMGDRModel
from src.models.lightgcn_backbone import build_normalized_adjacency
from src.metrics import cluster_probe_accuracy
from src.models.baselines import (
    MMGCNModel, LATTICEModel, BM3Model, FREEDOMModel, MGCNModel,
    LGMRecModel, MENTORModel, CausalRecModel, EliMRecModel,
)


CFG = load_config(PACKAGE_ROOT / "config" / "model.yaml")
SHARED = ROOT / "shared_data"
NEW = ensure_dir(ROOT / "new_results")


def build_loo(interactions):
    interactions = interactions.sort_values(["user_idx", "unixReviewTime"]).copy()
    interactions["split"] = "train"
    for user_idx, group in interactions.groupby("user_idx"):
        idx_list = group.index.tolist()
        if len(idx_list) >= 2:
            interactions.loc[idx_list[-1], "split"] = "test"
            interactions.loc[idx_list[-2], "split"] = "valid"
        elif len(idx_list) == 1:
            interactions.loc[idx_list[-1], "split"] = "test"
    return interactions


def prepare_runtime(seed=42):
    set_seed(seed)
    bundle = load_data_bundle(SHARED)
    loo = build_loo(bundle.interactions)
    bundle.interactions = loo
    train_inter = loo[loo["split"] == "train"]
    n_users = bundle.n_users
    src_arr = train_inter["user_idx"].values.astype(np.int64)
    dst_arr = (train_inter["item_idx"].values + n_users).astype(np.int64)
    row = np.concatenate([src_arr, dst_arr]); col = np.concatenate([dst_arr, src_arr])
    edge_index = torch.from_numpy(np.stack([row, col], 0))

    user_pos_all, user_pos_arrays = _build_user_positive_maps(loo)
    bundle.user_pos_all = user_pos_all
    bundle.user_pos_all_arrays = user_pos_arrays
    bundle.eval_splits = {
        "valid": _build_eval_split(loo, "valid"),
        "test": _build_eval_split(loo, "test"),
    }
    device = torch.device("cuda")
    norm_adj = build_normalized_adjacency(edge_index, bundle.n_nodes, device)

    visual_artifacts = prepare_visual_artifacts(CFG, PACKAGE_ROOT, bundle.n_items, seed)
    visual_features = torch.tensor(visual_artifacts.features, dtype=torch.float32, device=device)
    visual_clusters = torch.tensor(visual_artifacts.clusters, dtype=torch.long, device=device)
    cluster_prototypes = torch.tensor(visual_artifacts.prototypes, dtype=torch.float32, device=device)

    text_path = resolve_path(SHARED, CFG["text_feature_path"])
    text_features = torch.tensor(np.load(text_path).astype(np.float32), dtype=torch.float32, device=device)

    from scipy import sparse as sp
    ii_path = resolve_path(SHARED, CFG["item_item_graph_path"])
    ii_df = pd.read_parquet(ii_path)
    rr = np.concatenate([ii_df["item_idx_a"].values, ii_df["item_idx_b"].values])
    cc = np.concatenate([ii_df["item_idx_b"].values, ii_df["item_idx_a"].values])
    vv = np.ones(len(rr), dtype=np.float32)
    ii_coo = sp.coo_matrix((vv, (rr, cc)), shape=(bundle.n_items, bundle.n_items))
    ii_coo.sum_duplicates()
    deg = np.array(ii_coo.sum(axis=1)).flatten().clip(min=1.0)
    deg_inv_sqrt = np.power(deg, -0.5)
    ii_coo = sp.diags(deg_inv_sqrt) @ ii_coo @ sp.diags(deg_inv_sqrt)
    ii_coo = ii_coo.tocoo()
    ii_indices = torch.tensor(np.stack([ii_coo.row, ii_coo.col]), dtype=torch.long, device=device)
    ii_values = torch.tensor(ii_coo.data, dtype=torch.float32, device=device)
    item_item_adj = torch.sparse_coo_tensor(ii_indices, ii_values, (bundle.n_items, bundle.n_items)).coalesce()

    return {
        "bundle": bundle, "device": device, "norm_adj": norm_adj,
        "visual_features": visual_features, "visual_clusters": visual_clusters,
        "cluster_prototypes": cluster_prototypes, "text_features": text_features,
        "item_item_adj": item_item_adj, "visual_dim": visual_features.shape[1],
        "text_dim": text_features.shape[1], "n_clusters": int(visual_artifacts.summary["num_clusters"]),
    }


def _encode_cmgdr(ckpt_path: Path, mode: str, rt):
    state = torch.load(ckpt_path, map_location=rt["device"], weights_only=False)
    cfg_used = state.get("config", {})
    use_ii = cfg_used.get("use_item_item_graph", True)
    emb_dim = int(cfg_used.get("embedding_dim", state.get("embedding_dim", 256)))
    model = CMGDRModel(
        n_users=rt["bundle"].n_users, n_items=rt["bundle"].n_items,
        visual_dim=rt["visual_dim"], num_clusters=rt["n_clusters"],
        embedding_dim=emb_dim, num_layers=int(cfg_used.get("num_layers", 3)),
        text_dim=rt["text_dim"], use_item_item_graph=use_ii,
    ).to(rt["device"])
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        out = model.encode_all(
            norm_adj=rt["norm_adj"], visual_features=rt["visual_features"],
            visual_clusters=rt["visual_clusters"], cluster_prototypes=rt["cluster_prototypes"],
            mode=mode, grl_lambda=0.0,
            text_features=rt["text_features"],
            item_item_adj=rt["item_item_adj"] if use_ii else None,
        )
    return out["user_embeddings"], out["item_causal_embeddings"]


def _encode_baseline(ckpt_path: Path, model_class, rt, extra_kwargs=None):
    state = torch.load(ckpt_path, map_location=rt["device"], weights_only=False)
    emb_dim = int(state.get("embedding_dim", 256))
    kwargs = dict(
        n_users=rt["bundle"].n_users, n_items=rt["bundle"].n_items,
        visual_dim=rt["visual_dim"], text_dim=rt["text_dim"],
        embedding_dim=emb_dim, num_layers=3,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    model = model_class(**kwargs).to(rt["device"])
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        u, i = model.encode_all(rt["norm_adj"], rt["visual_features"], rt["text_features"])
    return u, i


def ipw_metrics(user_emb, item_emb, bundle, visual_clusters_np, cluster_prior, topk=10,
                n_negatives=999, seed=42, device="cuda"):
    """Compute (vanilla NDCG, IPW-NDCG, IPW-Recall, ESS) under sampled 1+999."""
    rng = np.random.default_rng(seed)
    test_split = bundle.eval_splits["test"]
    all_items_arr = np.arange(item_emb.size(0))

    # Compute weights w_v = (1/K) / P(V=v) — equivalent to Horvitz-Thompson under uniform
    K = len(cluster_prior)
    cluster_uniform = 1.0 / K
    w = cluster_uniform / np.clip(cluster_prior, 1e-8, None)
    # Normalize w so the average weight = 1 (stabilises variance)
    # use the empirical cluster distribution of test targets
    target_clusters = []
    for u, ts in test_split.targets_by_user.items():
        if len(ts):
            target_clusters.append(int(visual_clusters_np[int(ts[0])]))
    target_clusters = np.array(target_clusters, dtype=int)
    # No further normalisation — keep w as is so the estimator is unbiased

    rows = []
    n_users_done = 0
    for u, ts in test_split.targets_by_user.items():
        if len(ts) == 0:
            continue
        pos = int(ts[0])
        seen = bundle.user_pos_all.get(u, set())
        candidate_pool = np.setdiff1d(all_items_arr, np.fromiter(seen, dtype=np.int64), assume_unique=False)
        if len(candidate_pool) < n_negatives + 1:
            negs = candidate_pool[candidate_pool != pos].tolist()
        else:
            negs = rng.choice(candidate_pool, size=n_negatives, replace=False).tolist()
            negs = [n for n in negs if n != pos][:n_negatives]
        cands = np.array([pos] + list(negs), dtype=np.int64)
        ct = torch.tensor(cands, dtype=torch.long, device=device)
        scores = (user_emb[u].unsqueeze(0) * item_emb[ct]).sum(dim=-1)
        rank = int((scores > scores[0]).sum().item()) + 1
        ndcg = (1.0 / math.log2(rank + 1)) if rank <= topk else 0.0
        hit = 1.0 if rank <= topk else 0.0
        c = int(visual_clusters_np[pos])
        rows.append((ndcg, hit, w[c], 1.0 / rank))
        n_users_done += 1

    if not rows:
        return {}
    arr = np.array(rows)
    ndcg_v = arr[:, 0]; hit_v = arr[:, 1]; weight = arr[:, 2]; mrr_v = arr[:, 3]
    # Weighted (IPW) NDCG / Recall / MRR
    sw = weight.sum()
    ipw_ndcg = float((ndcg_v * weight).sum() / max(sw, 1e-9))
    ipw_recall = float((hit_v * weight).sum() / max(sw, 1e-9))
    ipw_mrr = float((mrr_v * weight).sum() / max(sw, 1e-9))
    # Effective sample size: (sum w)^2 / sum w^2 (Kish formula)
    ess = float(sw * sw / max((weight * weight).sum(), 1e-9))
    # Reweighting effect: vanilla vs IPW
    return {
        "n_users": int(len(rows)),
        "NDCG@10_vanilla": float(ndcg_v.mean()),
        "Recall@10_vanilla": float(hit_v.mean()),
        "MRR_vanilla": float(mrr_v.mean()),
        "NDCG@10_IPW": ipw_ndcg,
        "Recall@10_IPW": ipw_recall,
        "MRR_IPW": ipw_mrr,
        "delta_NDCG_vs_vanilla": ipw_ndcg - float(ndcg_v.mean()),
        "delta_Recall_vs_vanilla": ipw_recall - float(hit_v.mean()),
        "ESS": ess,
        "ESS_ratio": ess / max(len(rows), 1),
    }


def main():
    rt = prepare_runtime(seed=42)
    bundle = rt["bundle"]
    visual_clusters_np = rt["visual_clusters"].cpu().numpy()
    K = rt["n_clusters"]

    # Catalogue cluster prior P(V=v) — from the entire item set
    cluster_prior = np.bincount(visual_clusters_np, minlength=K).astype(np.float64)
    cluster_prior /= cluster_prior.sum()
    print(f"Cluster prior (K={K}): min={cluster_prior.min():.4f}, max={cluster_prior.max():.4f}, "
          f"std={cluster_prior.std():.4f}")

    methods = []
    ckpt_dir = SHARED / "model_outputs" / "checkpoints"
    for name, cls, extra in [
        ("MMGCN", MMGCNModel, {}),
        ("LATTICE", LATTICEModel, {"k_neighbors": 10}),
        ("BM3", BM3Model, {"cl_weight": 0.1, "temperature": 0.2, "dropout": 0.3}),
        ("FREEDOM", FREEDOMModel, {"k_neighbors": 10}),
        ("MGCN", MGCNModel, {}),
        ("LGMRec", LGMRecModel, {}),
        ("MENTOR", MENTORModel, {}),
        ("CausalRec", CausalRecModel, {}),
        ("EliMRec", EliMRecModel, {}),
    ]:
        p = ckpt_dir / f"{name}_seed42.pt"
        if p.exists():
            methods.append((name, p, "baseline", cls, extra))

    for name, mode in [
        ("MM-LightGCN", "lightgcn"),
        ("CMGDR-Residual", "residual"),
        ("CMGDR-Full", "full"),
    ]:
        for fname in [f"{name}_seed42_seed42.pt", f"{name}_seed42.pt"]:
            p = ckpt_dir / fname
            if p.exists():
                methods.append((name, p, "cmgdr", mode, None))
                break

    print(f"\nMethods found ({len(methods)}): {[m[0] for m in methods]}")

    rows = []
    for entry in methods:
        name, ckpt, kind = entry[0], entry[1], entry[2]
        t0 = time.time()
        if kind == "baseline":
            user_emb, item_emb = _encode_baseline(ckpt, entry[3], rt, extra_kwargs=entry[4])
        else:
            user_emb, item_emb = _encode_cmgdr(ckpt, entry[3], rt)
        m = ipw_metrics(user_emb, item_emb, bundle, visual_clusters_np, cluster_prior,
                        topk=10, n_negatives=999, seed=42, device=rt["device"])
        m["method"] = name
        m["checkpoint"] = str(ckpt.relative_to(ROOT))
        m["encode_time_sec"] = round(time.time() - t0, 1)
        rows.append(m)
        print(f"  {name:<14} NDCG@10[vanilla]={m['NDCG@10_vanilla']:.4f}  "
              f"NDCG@10[IPW]={m['NDCG@10_IPW']:.4f}  Δ={m['delta_NDCG_vs_vanilla']:+.4f}  "
              f"ESS={m['ESS']:.0f}/{m['n_users']}")

    save_json(NEW / "ipw_dcg.json", rows)
    pd.DataFrame(rows).to_csv(NEW / "ipw_dcg.csv", index=False)
    print(f"\nSaved: {NEW / 'ipw_dcg.json'}")


if __name__ == "__main__":
    main()
