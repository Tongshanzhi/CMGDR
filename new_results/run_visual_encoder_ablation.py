"""Section 7.4 — Visual encoder ablation.

Re-extract item visual features with 2 alternative encoders:
  (i)  CLIP-ViT-B/32  (transformers, 512-d)
  (ii) ResNet-50      (torchvision, 2048-d)
Re-cluster (KMeans, K=32, seed=42) and re-train CMGDR-Full at seed 42 against each.

Compare to the headline ViT-B/16 result (already in shared_data).
The Probe→accuracy story is encoder-agnostic if the residual decomposition
keeps Probe < LightGCN baseline and Recall@10 > MM-LightGCN across encoders.
"""
from __future__ import annotations

import json
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module_B"))

from src.utils import PACKAGE_ROOT, load_config, ensure_dir, save_json
from run_full_evaluation import run_experiment as run_cmgdr_experiment


CFG = load_config(PACKAGE_ROOT / "config" / "model.yaml")
NEW = ensure_dir(ROOT / "new_results")
SHARED = ROOT / "shared_data"
IMG_DIR = SHARED / "images"
PROCESSED = SHARED / "processed"


# --------------------------------------------------------------------------
class ItemImageDataset(Dataset):
    def __init__(self, image_dir: Path, n_items: int, transform, missing: set):
        self.image_dir = image_dir
        self.n_items = n_items
        self.transform = transform
        self.missing = missing

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        if idx in self.missing:
            return idx, torch.zeros(3, 224, 224), False
        p = self.image_dir / f"{idx}.jpg"
        if not p.exists():
            return idx, torch.zeros(3, 224, 224), False
        try:
            img = Image.open(p).convert("RGB")
            return idx, self.transform(img), True
        except Exception:
            return idx, torch.zeros(3, 224, 224), False


# --------------------------------------------------------------------------
def _missing_idxs(n_items: int):
    manifest = json.loads((IMG_DIR / "download_manifest.json").read_text())
    miss = set(manifest.get("no_url_item_idxs", []))
    for entry in manifest.get("failed", []):
        miss.add(entry["item_idx"])
    # Also mark anything not on disk
    for i in range(n_items):
        if i in miss:
            continue
        if not (IMG_DIR / f"{i}.jpg").exists():
            miss.add(i)
    return miss


# --------------------------------------------------------------------------
def extract_resnet50(n_items: int, device: str = "cuda"):
    out_path = PROCESSED / "item_visual_embeddings_resnet50.npy"
    if out_path.exists():
        print(f"  [skip] {out_path} exists")
        return out_path
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
    model = tv_models.resnet50(weights=weights)
    visual_dim = model.fc.in_features  # 2048
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    miss = _missing_idxs(n_items)
    ds = ItemImageDataset(IMG_DIR, n_items, transform, miss)
    loader = DataLoader(ds, batch_size=128, num_workers=4, pin_memory=True)
    emb = np.zeros((n_items, visual_dim), dtype=np.float32)
    valid = np.zeros(n_items, dtype=bool)
    with torch.no_grad():
        for idxs, batch, ok in tqdm(loader, desc="ResNet50"):
            batch = batch.to(device, non_blocking=True)
            f = model(batch).cpu().numpy().astype(np.float32)
            for j, ix in enumerate(idxs.tolist()):
                emb[ix] = f[j]
                valid[ix] = bool(ok[j])
    # Normalize valid rows
    vi = np.where(valid)[0]
    norms = np.linalg.norm(emb[vi], axis=1, keepdims=True).clip(min=1e-8)
    emb[vi] = emb[vi] / norms
    # Fill missing with mean
    inv = np.where(~valid)[0]
    if len(vi) and len(inv):
        emb[inv] = emb[vi].mean(axis=0)
    np.save(out_path, emb)
    print(f"  ResNet50 saved: {emb.shape} -> {out_path} (valid={int(valid.sum())})")
    return out_path


def extract_clip_b32(n_items: int, device: str = "cuda"):
    out_path = PROCESSED / "item_visual_embeddings_clip_b32.npy"
    if out_path.exists():
        print(f"  [skip] {out_path} exists")
        return out_path
    from transformers import CLIPModel, CLIPProcessor
    model_name = "openai/clip-vit-base-patch32"
    print(f"  loading CLIP {model_name} (use_safetensors=True)")
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    visual_dim = int(model.config.projection_dim)  # 512
    miss = _missing_idxs(n_items)

    # CLIP processor takes raw PIL — bypass torchvision
    class _CLIPDataset(Dataset):
        def __init__(self):
            pass

        def __len__(self):
            return n_items

        def __getitem__(self, idx):
            if idx in miss:
                return idx, np.zeros((224, 224, 3), dtype=np.uint8), False
            p = IMG_DIR / f"{idx}.jpg"
            if not p.exists():
                return idx, np.zeros((224, 224, 3), dtype=np.uint8), False
            try:
                img = Image.open(p).convert("RGB").resize((224, 224))
                return idx, np.asarray(img, dtype=np.uint8), True
            except Exception:
                return idx, np.zeros((224, 224, 3), dtype=np.uint8), False

    def _collate(batch):
        idxs = [b[0] for b in batch]
        imgs = [Image.fromarray(b[1]) for b in batch]
        oks = [b[2] for b in batch]
        return idxs, imgs, oks

    loader = DataLoader(_CLIPDataset(), batch_size=64, num_workers=4, collate_fn=_collate)
    emb = np.zeros((n_items, visual_dim), dtype=np.float32)
    valid = np.zeros(n_items, dtype=bool)
    with torch.no_grad():
        for idxs, imgs, oks in tqdm(loader, desc="CLIP-ViT-B/32"):
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)
            # transformers >=5 returns BaseModelOutputWithPooling; .pooler_output holds the projected (512-d) embedding
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            f = feats.cpu().numpy().astype(np.float32)
            for j, ix in enumerate(idxs):
                emb[ix] = f[j]
                valid[ix] = bool(oks[j])
    vi = np.where(valid)[0]
    norms = np.linalg.norm(emb[vi], axis=1, keepdims=True).clip(min=1e-8)
    emb[vi] = emb[vi] / norms
    inv = np.where(~valid)[0]
    if len(vi) and len(inv):
        emb[inv] = emb[vi].mean(axis=0)
    np.save(out_path, emb)
    print(f"  CLIP saved: {emb.shape} -> {out_path} (valid={int(valid.sum())})")
    return out_path


# --------------------------------------------------------------------------
def cluster_features(emb_path: Path, K: int = 32, seed: int = 42):
    out = emb_path.with_name(emb_path.stem.replace("item_visual_embeddings_", "item_visual_clusters_") + ".csv")
    if out.exists():
        print(f"  [skip] {out} exists")
        return out
    emb = np.load(emb_path)
    print(f"  KMeans K={K} on {emb.shape} (seed={seed})")
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    cl = km.fit_predict(emb).astype(np.int64)
    import pandas as pd
    pd.DataFrame({"item_idx": np.arange(len(cl)), "cluster_id": cl}).to_csv(out, index=False)
    print(f"  Clusters -> {out}")
    return out


def train_cmgdr(encoder_name: str, feat_path: Path, cl_path: Path):
    """Run CMGDR-Full with the new visual feature/cluster files."""
    overrides = {
        "use_item_item_graph": True,
        "embedding_dim": 256,
        "learning_rate": 0.002,
        "visual_feature_path": str(feat_path),
        "visual_cluster_path": str(cl_path),
        "loss_weights": {
            "residual": 0.5, "adversarial": 0.01, "counterfactual": 0.5,
            "orthogonality": 0.05, "contrastive": 0,
            "contrastive_temperature": 0.2, "text_consistency": 0,
        },
    }
    t0 = time.time()
    out = run_cmgdr_experiment(
        config=CFG, mode="full", seed=42,
        suffix=f"CMGDR-Full_{encoder_name}", num_epochs=40, overrides=overrides,
    )
    out.update({
        "experiment": "visual_encoder_ablation",
        "encoder": encoder_name,
        "visual_feature_path": str(feat_path),
        "visual_cluster_path": str(cl_path),
        "time_sec": time.time() - t0,
        "seed": 42,
    })
    return out


def main():
    n_items = 18357
    device = "cuda"
    results = []
    log = NEW / "visual_encoder_log.jsonl"
    log.unlink(missing_ok=True)

    print("\n========== VISUAL ENCODER ABLATION ==========")
    # 1) extract features
    print("\n[1/3] Extract ResNet-50")
    rn_path = extract_resnet50(n_items, device)
    print("\n[2/3] Extract CLIP-ViT-B/32")
    clip_path = extract_clip_b32(n_items, device)

    # 2) cluster
    print("\n[3/3] Cluster K=32 for each")
    rn_cl = cluster_features(rn_path)
    clip_cl = cluster_features(clip_path)

    # 3) train CMGDR-Full per encoder
    for name, fp, cp in [("resnet50", rn_path, rn_cl), ("clip_b32", clip_path, clip_cl)]:
        print(f"\n>>> Training CMGDR-Full with {name}")
        r = train_cmgdr(name, fp, cp)
        results.append(r)
        with open(log, "a") as f:
            f.write(json.dumps(r, default=float) + "\n")
        print(f"  {name}: R@10={r.get('Recall@10', 0):.4f} N@10={r.get('NDCG@10', 0):.4f} "
              f"Probe_c={r.get('probe_causal', 0):.3f} CFShift={r.get('cf_score_shift', 0):.4f}")

    save_json(NEW / "visual_encoder_raw.json", results)
    print(f"\nSaved: {NEW / 'visual_encoder_raw.json'}")


if __name__ == "__main__":
    main()
