"""Sampled evaluation protocol (1 positive + 999 negatives) for paper-comparable metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .data import DataBundle


def sampled_evaluate(
    model,
    norm_adj: torch.Tensor,
    visual_features: torch.Tensor,
    visual_clusters: torch.Tensor,
    cluster_prototypes: torch.Tensor,
    bundle: DataBundle,
    device: torch.device,
    mode: str,
    n_negatives: int = 999,
    topk_list: list[int] | None = None,
    seed: int = 42,
    text_features: torch.Tensor | None = None,
    item_item_adj: torch.Tensor | None = None,
    split_name: str = "test",
) -> dict[str, Any]:
    """
    For each test user, sample 1 positive + n_negatives random negatives,
    rank among these (n_negatives+1) items. This matches the protocol used
    in MMGCN, LATTICE, BM3, FREEDOM, etc.
    """
    if topk_list is None:
        topk_list = [10, 20]

    rng = np.random.default_rng(seed)
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

    user_emb = outputs["user_embeddings"]
    item_causal_emb = outputs["item_causal_embeddings"]

    all_items = set(range(bundle.n_items))
    metrics_per_user: list[dict[str, float]] = []

    for user_idx in split.user_ids:
        user_idx = int(user_idx)
        targets = split.targets_by_user[user_idx]
        if len(targets) == 0:
            continue

        # Pick one positive item (the first target)
        pos_item = int(targets[0])

        # Sample negatives: items not in ANY of this user's interactions
        user_all_pos = bundle.user_pos_all.get(user_idx, set())
        neg_pool = list(all_items - user_all_pos)
        if len(neg_pool) < n_negatives:
            neg_items = neg_pool
        else:
            neg_items = rng.choice(neg_pool, size=n_negatives, replace=False).tolist()

        # Candidate set: 1 positive + n_negatives
        candidates = [pos_item] + neg_items
        candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

        # Score
        u_emb = user_emb[user_idx].unsqueeze(0)  # (1, dim)
        c_emb = item_causal_emb[candidate_tensor]  # (1000, dim)
        scores = (u_emb * c_emb).sum(dim=-1)  # (1000,)

        # Rank: the positive item is at index 0
        rank_of_pos = int((scores > scores[0]).sum().item()) + 1  # 1-based rank

        user_metrics: dict[str, float] = {}
        for k in topk_list:
            hit = 1.0 if rank_of_pos <= k else 0.0
            user_metrics[f"HR@{k}"] = hit
            user_metrics[f"Recall@{k}"] = hit  # with 1 positive, Recall@K = HR@K
            user_metrics[f"NDCG@{k}"] = (1.0 / math.log2(rank_of_pos + 1)) if rank_of_pos <= k else 0.0
        user_metrics["MRR"] = 1.0 / rank_of_pos
        metrics_per_user.append(user_metrics)

    # Aggregate
    if not metrics_per_user:
        return {"n_eval_users": 0}

    keys = sorted(metrics_per_user[0].keys())
    summary = {key: float(np.mean([m[key] for m in metrics_per_user])) for key in keys}
    summary["n_eval_users"] = len(metrics_per_user)
    summary["n_negatives"] = n_negatives
    summary["protocol"] = f"sampled_1+{n_negatives}"
    summary["mode"] = mode
    return summary
