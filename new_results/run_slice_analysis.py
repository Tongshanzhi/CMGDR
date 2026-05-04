"""Section 7.6 — Long-tail / cold-start slice analysis.

Sampled-1+999 evaluation restricted to:
  (a) long-tail items: bottom 10% of items by training popularity
  (b) cold-start users: users with <= 10 total interactions
For each method whose checkpoint is on disk, recompute Recall@10, NDCG@10
and MRR. Probe is the catalogue-wide visual cluster probe (unchanged).
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
    text_np = np.load(text_path).astype(np.float32)
    text_features = torch.tensor(text_np, dtype=torch.float32, device=device)

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


def sampled_metric_subpool(user_emb, item_emb, bundle, eligible_users, item_pool: np.ndarray,
                           topk=10, n_negatives=999, seed=42, device="cuda",
                           require_target_in_pool=True):
    """For each user in eligible_users:
       - take 1 positive from this user's test targets that lies in item_pool (if require_target_in_pool)
       - draw n_negatives random negatives from item_pool, excluding user's positives
       - rank, compute hit/ndcg/mrr.
    """
    rng = np.random.default_rng(seed)
    test_split = bundle.eval_splits["test"]
    pool_set = set(item_pool.tolist())
    rows = []
    for u in eligible_users:
        targets = test_split.targets_by_user.get(u, np.array([]))
        if len(targets) == 0:
            continue
        if require_target_in_pool:
            in_pool_targets = [int(t) for t in targets.tolist() if int(t) in pool_set]
            if not in_pool_targets:
                continue
            pos = int(in_pool_targets[0])
        else:
            pos = int(targets[0])
        seen = bundle.user_pos_all.get(u, set())
        # negatives drawn from pool, excluding `seen`
        candidate_pool = np.setdiff1d(item_pool, np.fromiter(seen, dtype=np.int64), assume_unique=False)
        if len(candidate_pool) < n_negatives + 1:
            negs = candidate_pool[candidate_pool != pos].tolist()
        else:
            negs = rng.choice(candidate_pool, size=n_negatives, replace=False).tolist()
            negs = [n for n in negs if n != pos][:n_negatives]
        cands = np.array([pos] + list(negs), dtype=np.int64)
        ct = torch.tensor(cands, dtype=torch.long, device=device)
        scores = (user_emb[u].unsqueeze(0) * item_emb[ct]).sum(dim=-1)
        rank = int((scores > scores[0]).sum().item()) + 1
        hit10 = 1.0 if rank <= topk else 0.0
        ndcg = (1.0 / math.log2(rank + 1)) if rank <= topk else 0.0
        mrr = 1.0 / rank
        rows.append((hit10, ndcg, mrr))

    if not rows:
        return {"n_users": 0, "Recall@10": 0.0, "NDCG@10": 0.0, "MRR": 0.0}
    arr = np.array(rows)
    return {
        "n_users": len(rows),
        "Recall@10": float(arr[:, 0].mean()),
        "NDCG@10": float(arr[:, 1].mean()),
        "MRR": float(arr[:, 2].mean()),
    }


def main():
    rt = prepare_runtime(seed=42)
    bundle = rt["bundle"]
    interactions = bundle.interactions
    train = interactions[interactions["split"] == "train"]
    item_pop = train.groupby("item_idx").size().reindex(range(bundle.n_items), fill_value=0)
    pop = item_pop.values

    cutoff = np.percentile(pop, 10)
    lt_items = np.where(pop <= cutoff)[0]
    print(f"Long-tail (bottom-decile) cutoff: pop<={cutoff:.0f}, n_items_in_slice={len(lt_items)}")

    all_items = np.arange(bundle.n_items)

    user_n_total = interactions.groupby("user_idx").size()
    cold_users = set(user_n_total[user_n_total <= 10].index.tolist())
    test_users = sorted(bundle.eval_splits["test"].targets_by_user.keys())
    cold_eligible = sorted(set(test_users).intersection(cold_users))
    print(f"Cold-start (<=10 interactions) users: {len(cold_users)}; on test: {len(cold_eligible)}")

    methods = []  # list of (name, ckpt, kind, mode_or_class, extra)
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

    # CMGDR variants — saved by run_multiseed.py with suffix _seed{seed}, then run_full_evaluation appends another _seed42
    for name, mode in [
        ("MM-LightGCN", "lightgcn"),
        ("CMGDR-Residual", "residual"),
        ("CMGDR-Full", "full"),
    ]:
        # multiseed produces e.g. CMGDR-Full_seed42_seed42.pt
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

        # Full pool baseline
        full_res = sampled_metric_subpool(user_emb, item_emb, bundle, test_users,
                                          all_items, topk=10, n_negatives=999, seed=42,
                                          device=rt["device"], require_target_in_pool=False)
        # Long-tail pool: target must be in long-tail; negatives from long-tail
        lt_res = sampled_metric_subpool(user_emb, item_emb, bundle, test_users,
                                        lt_items, topk=10, n_negatives=999, seed=42,
                                        device=rt["device"], require_target_in_pool=True)
        # Cold-start: full item pool but only cold users
        cs_res = sampled_metric_subpool(user_emb, item_emb, bundle, cold_eligible,
                                        all_items, topk=10, n_negatives=999, seed=42,
                                        device=rt["device"], require_target_in_pool=False)

        # Probe over the entire item embedding (catalogue-wide)
        probe = cluster_probe_accuracy(item_emb.cpu().numpy(), rt["visual_clusters"].cpu().numpy())

        row = {
            "method": name,
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "Recall@10_full": full_res["Recall@10"],
            "NDCG@10_full": full_res["NDCG@10"],
            "MRR_full": full_res["MRR"],
            "n_users_full": full_res["n_users"],
            "Recall@10_longtail": lt_res["Recall@10"],
            "NDCG@10_longtail": lt_res["NDCG@10"],
            "MRR_longtail": lt_res["MRR"],
            "n_users_longtail": lt_res["n_users"],
            "Recall@10_cold": cs_res["Recall@10"],
            "NDCG@10_cold": cs_res["NDCG@10"],
            "MRR_cold": cs_res["MRR"],
            "n_users_cold": cs_res["n_users"],
            "probe_item_emb": probe,
            "encode_time_sec": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"  {name:<14} R@10[full]={row['Recall@10_full']:.4f} "
              f"R@10[long-tail]={row['Recall@10_longtail']:.4f}  "
              f"R@10[cold]={row['Recall@10_cold']:.4f}  Probe={probe:.3f}")

    save_json(NEW / "slice_analysis.json", rows)
    pd.DataFrame(rows).to_csv(NEW / "slice_analysis.csv", index=False)
    print(f"\nSaved: {NEW / 'slice_analysis.json'}")


if __name__ == "__main__":
    main()
