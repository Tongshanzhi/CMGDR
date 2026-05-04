"""Section 7.5 — Stronger counter-factuals via Stable Diffusion.

Replace the cluster-prototype intervention with an actual image perturbation:
1. Sample N items from the test set.
2. Load each item's original image.
3. Run SD-Turbo img2img with a generic style prompt (strength=0.5) → CF image.
4. Re-encode the CF image with ViT-B/16 → cf_visual_features[item_idx].
5. Run CMGDR-Full's encode_all twice (original vs CF visual features).
6. Compute CF-Shift = mean_users mean_items |score_orig - score_cf| for sampled items.
   Compare to the prototype-based CF-Shift = 0.000 reported in the paper.

Real data: original Sports & Outdoors images + real SD generation + ViT re-encoding +
CMGDR-Full's actual encoder. No mocks.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module_B"))

from src.utils import PACKAGE_ROOT, load_config, ensure_dir, save_json, set_seed, resolve_path
from src.data import load_data_bundle, _build_eval_split, _build_user_positive_maps
from src.features import prepare_visual_artifacts
from src.models.causal_debias import CMGDRModel
from src.models.lightgcn_backbone import build_normalized_adjacency

SHARED = ROOT / "shared_data"
NEW = ensure_dir(ROOT / "new_results")
IMG_DIR = SHARED / "images"
CKPT = SHARED / "model_outputs" / "checkpoints" / "CMGDR-Full_seed42_seed42.pt"

CFG = load_config(PACKAGE_ROOT / "config" / "model.yaml")


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

    user_pos_all, _ = _build_user_positive_maps(loo)
    bundle.user_pos_all = user_pos_all
    bundle.eval_splits = {
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
    import pandas as pd
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


def load_model(rt):
    state = torch.load(CKPT, map_location=rt["device"], weights_only=False)
    cfg_used = state.get("config", {})
    use_ii = cfg_used.get("use_item_item_graph", True)
    emb_dim = int(cfg_used.get("embedding_dim", 256))
    model = CMGDRModel(
        n_users=rt["bundle"].n_users, n_items=rt["bundle"].n_items,
        visual_dim=rt["visual_dim"], num_clusters=rt["n_clusters"],
        embedding_dim=emb_dim, num_layers=int(cfg_used.get("num_layers", 3)),
        text_dim=rt["text_dim"], use_item_item_graph=use_ii,
    ).to(rt["device"])
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def encode(model, rt, visual_features):
    out = model.encode_all(
        norm_adj=rt["norm_adj"], visual_features=visual_features,
        visual_clusters=rt["visual_clusters"], cluster_prototypes=rt["cluster_prototypes"],
        mode="full", grl_lambda=0.0,
        text_features=rt["text_features"],
        item_item_adj=rt["item_item_adj"],
    )
    return out["user_embeddings"], out["item_total_embeddings"]


def vit_extract(images, vit, transform, device):
    """Extract ViT features for a list of PIL images. Returns (n, 768) normalized."""
    tensors = torch.stack([transform(img) for img in images]).to(device)
    with torch.no_grad():
        feats = vit(tensors).cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-8)
    return feats / norms


def main(n_sample=200, seed=42, strength=0.5, num_inference_steps=2, prompt=None):
    set_seed(seed)
    rng = np.random.default_rng(seed)

    rt = prepare_runtime(seed=seed)
    bundle = rt["bundle"]
    n_items = bundle.n_items

    # Sample items: those that (a) appear as a test target and (b) have an image on disk
    test_items = set()
    for ts in bundle.eval_splits["test"].targets_by_user.values():
        for t in ts:
            test_items.add(int(t))
    test_items = list(test_items)
    rng.shuffle(test_items)
    sampled_idxs = []
    for i in test_items:
        if (IMG_DIR / f"{i}.jpg").exists():
            sampled_idxs.append(i)
        if len(sampled_idxs) >= n_sample:
            break
    sampled_idxs = np.array(sampled_idxs, dtype=np.int64)
    print(f"Sampled {len(sampled_idxs)} items from test pool")

    # Load CMGDR-Full
    print(f"Loading {CKPT.name}")
    model = load_model(rt)

    # Build ViT for re-encoding
    vit_w = tv_models.ViT_B_16_Weights.IMAGENET1K_V1
    vit = tv_models.vit_b_16(weights=vit_w)
    vit.heads = torch.nn.Identity()
    vit = vit.to(rt["device"]).eval()
    for p in vit.parameters():
        p.requires_grad = False
    vit_transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load SD-Turbo img2img
    from diffusers import AutoPipelineForImage2Image
    print("Loading SD-Turbo")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16",
        cache_dir=os.environ["HF_HOME"], safety_checker=None,
    ).to(rt["device"])
    pipe.set_progress_bar_config(disable=True)

    # Default prompt: a generic style transfer that should perturb visual cluster
    # without destroying product identity
    if prompt is None:
        prompt = "a high-quality product photograph in a different lighting and background, professional e-commerce style"

    # ---- Generate CF images and re-extract ViT features ----
    cf_visual = rt["visual_features"].clone()
    successes = []
    t0 = time.time()
    batch = 8
    for k in tqdm(range(0, len(sampled_idxs), batch), desc="SD img2img"):
        chunk = sampled_idxs[k:k+batch]
        # load originals
        imgs = []
        keep = []
        for i in chunk:
            p = IMG_DIR / f"{int(i)}.jpg"
            try:
                img = Image.open(p).convert("RGB").resize((512, 512))
                imgs.append(img)
                keep.append(int(i))
            except Exception as e:
                print(f"  skip {i}: {e}")
        if not imgs:
            continue
        try:
            # SD-Turbo img2img: 1-2 inference steps, no guidance
            generator = torch.Generator(device=rt["device"]).manual_seed(seed + int(keep[0]))
            cf_imgs = pipe(
                prompt=[prompt] * len(imgs), image=imgs,
                num_inference_steps=num_inference_steps,
                strength=strength, guidance_scale=0.0,
                generator=generator,
            ).images
        except Exception as e:
            print(f"  SD failed on batch {k}: {e}")
            continue
        # re-encode with ViT-B/16
        feats = vit_extract(cf_imgs, vit, vit_transform, rt["device"])
        for ii, idx in enumerate(keep):
            cf_visual[idx] = torch.tensor(feats[ii], device=rt["device"])
            successes.append(idx)
    print(f"Generated CF for {len(successes)} items in {time.time()-t0:.1f}s")

    # ---- Encode original and CF; compute CF-Shift over all (user, sampled_item) pairs ----
    user_emb, item_total_orig = encode(model, rt, rt["visual_features"])
    _, item_total_cf = encode(model, rt, cf_visual)

    successes = np.array(successes, dtype=np.int64)
    s_t = torch.tensor(successes, dtype=torch.long, device=rt["device"])
    # user-item scores for sampled items
    scores_orig = user_emb @ item_total_orig[s_t].T  # (n_users, n_sampled)
    scores_cf = user_emb @ item_total_cf[s_t].T
    abs_shift = (scores_orig - scores_cf).abs().mean().item()
    item_norm_shift = (item_total_orig[s_t] - item_total_cf[s_t]).norm(dim=-1).mean().item()
    item_relative_shift = ((item_total_orig[s_t] - item_total_cf[s_t]).norm(dim=-1)
                           / item_total_orig[s_t].norm(dim=-1).clamp(min=1e-8)).mean().item()

    # Visual feature drift caused by SD
    feat_orig = rt["visual_features"][s_t]
    feat_cf = cf_visual[s_t]
    cos_sim = torch.nn.functional.cosine_similarity(feat_orig, feat_cf, dim=-1).mean().item()
    visual_l2 = (feat_orig - feat_cf).norm(dim=-1).mean().item()

    # Also rank shift: how often does CF change top-10?
    # For computational simplicity, sample 256 random users and look at top-K shift
    sample_users = rng.choice(rt["bundle"].n_users, size=min(256, rt["bundle"].n_users), replace=False)
    sample_users_t = torch.tensor(sample_users, dtype=torch.long, device=rt["device"])
    full_orig = user_emb[sample_users_t] @ item_total_orig.T  # (256, n_items)
    full_cf = user_emb[sample_users_t] @ item_total_cf.T
    # rank of each sampled item BEFORE/AFTER CF, in each user's ranking
    rank_shifts = []
    for col, idx in enumerate(successes):
        # Rank of item `idx` for each sampled user
        sc_o = full_orig[:, idx]
        sc_c = full_cf[:, idx]
        rank_o = (full_orig > sc_o.unsqueeze(1)).sum(dim=1)
        rank_c = (full_cf > sc_c.unsqueeze(1)).sum(dim=1)
        rank_shifts.append((rank_c - rank_o).abs().float().mean().item())
    mean_rank_shift = float(np.mean(rank_shifts)) if rank_shifts else 0.0

    result = {
        "method": "CMGDR-Full",
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "n_items_sampled": int(len(successes)),
        "sd_model": "stabilityai/sd-turbo",
        "sd_strength": strength,
        "sd_steps": num_inference_steps,
        "sd_prompt": prompt,
        "abs_score_shift": abs_shift,
        "item_emb_l2_shift": item_norm_shift,
        "item_emb_relative_l2_shift": item_relative_shift,
        "visual_feature_cosine_similarity": cos_sim,
        "visual_feature_l2_drift": visual_l2,
        "mean_abs_rank_shift_over_sample_users": mean_rank_shift,
        "n_sample_users_for_rank": int(len(sample_users)),
        "comparison_prototype_based_cf_shift": 0.0003,  # CMGDR-Full from paper (Section 7.9 sees 0.0003)
    }
    save_json(NEW / "sd_counterfactual.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sample", type=int, default=200)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()
    main(n_sample=args.n_sample, strength=args.strength, num_inference_steps=args.steps)
