"""Extract visual embeddings from downloaded images using a pretrained ViT/CNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(model_name: str, weights: str, device: torch.device):
    """Load a torchvision model, remove the classification head, return (model, visual_dim)."""
    if model_name == "vit_b_16":
        w = models.ViT_B_16_Weights[weights]
        base = models.vit_b_16(weights=w)
        visual_dim = base.heads.head.in_features  # 768
        base.heads = torch.nn.Identity()
    elif model_name == "resnet50":
        w = models.ResNet50_Weights[weights]
        base = models.resnet50(weights=w)
        visual_dim = base.fc.in_features  # 2048
        base.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    base = base.to(device).eval()
    for p in base.parameters():
        p.requires_grad = False
    return base, visual_dim


class ItemImageDataset(Dataset):
    """Yields (item_idx, image_tensor) for every item in [0, n_items).
    Missing images get a zero tensor and are flagged."""

    def __init__(self, image_dir: Path, n_items: int, transform, missing_idxs: set[int]):
        self.image_dir = image_dir
        self.n_items = n_items
        self.transform = transform
        self.missing_idxs = missing_idxs

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int):
        if idx in self.missing_idxs:
            return idx, torch.zeros(3, 224, 224), False

        path = self.image_dir / f"{idx}.jpg"
        if not path.exists():
            return idx, torch.zeros(3, 224, 224), False

        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return idx, tensor, True
        except Exception:
            return idx, torch.zeros(3, 224, 224), False


def run_extract(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root)
    cfg = _load_config(Path(config_path) if config_path else root / "config" / "visual.yaml")
    ext = cfg["extract"]
    up = cfg["upstream"]
    dl_cfg = cfg["download"]

    device = _choose_device(ext.get("device", "auto"))
    print(f"Using device: {device}")

    # Load n_items
    data_root = Path(up["data_root"])
    with open(data_root / up["item_id_map"], encoding="utf-8") as f:
        n_items = len(json.load(f))
    print(f"Total items: {n_items}")

    # Load download manifest to know which items are missing
    manifest_path = root / dl_cfg["manifest_path"]
    missing_idxs: set[int] = set()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing_idxs = set(manifest.get("no_url_item_idxs", []))
        for entry in manifest.get("failed", []):
            missing_idxs.add(entry["item_idx"])
    print(f"Missing images (no URL + failed): {len(missing_idxs)}")

    # Build model
    model, visual_dim = _build_model(ext["model_name"], ext["weights"], device)
    print(f"Model: {ext['model_name']}, visual_dim: {visual_dim}")

    # Image transform
    img_size = ext.get("image_size", 224)
    transform = T.Compose([
        T.Resize(img_size + 32),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset & DataLoader
    image_dir = root / dl_cfg["output_dir"]
    dataset = ItemImageDataset(image_dir, n_items, transform, missing_idxs)
    loader = DataLoader(
        dataset,
        batch_size=ext.get("batch_size", 128),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Extract
    embeddings = np.zeros((n_items, visual_dim), dtype=np.float32)
    valid_mask = np.zeros(n_items, dtype=bool)

    with torch.no_grad():
        for batch_idx, batch_tensor, batch_valid in tqdm(loader, desc="Extracting features"):
            batch_tensor = batch_tensor.to(device, non_blocking=True)
            features = model(batch_tensor)  # (B, visual_dim)
            features = features.cpu().numpy().astype(np.float32)

            for i, idx in enumerate(batch_idx.tolist()):
                embeddings[idx] = features[i]
                valid_mask[idx] = bool(batch_valid[i])

    # Normalize valid embeddings
    if ext.get("normalize", True):
        valid_indices = np.where(valid_mask)[0]
        norms = np.linalg.norm(embeddings[valid_indices], axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings[valid_indices] = embeddings[valid_indices] / norms

    # Fill strategy for missing items
    fill_strategy = ext.get("fill_strategy", "zero")
    invalid_indices = np.where(~valid_mask)[0]
    if fill_strategy == "mean" and len(invalid_indices) > 0:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            mean_vec = embeddings[valid_indices].mean(axis=0)
            embeddings[invalid_indices] = mean_vec
            print(f"Filled {len(invalid_indices)} missing items with mean vector")
    else:
        print(f"Missing items ({len(invalid_indices)}) filled with zero vectors")

    # Save
    output_path = root / ext["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    summary = {
        "n_items": n_items,
        "visual_dim": visual_dim,
        "model_name": ext["model_name"],
        "weights": ext["weights"],
        "n_valid": int(valid_mask.sum()),
        "n_missing": int((~valid_mask).sum()),
        "fill_strategy": fill_strategy,
        "normalized": ext.get("normalize", True),
        "output_path": str(output_path),
        "embedding_shape": list(embeddings.shape),
    }
    print(f"Saved embeddings: {embeddings.shape} -> {output_path}")
    return summary
