#!/usr/bin/env python3
"""
Config-driven ReID trainer.

Usage:
    python train.py --config configs/experiments/exp03_baseline_resnet50_cross.yaml
    python train.py --config <path> --max_samples 500 --epochs 2   # debug
    python train.py --config <path> --resume checkpoints/<exp_id>/last.pth

Features:
- YAML configs (see configs/experiments/*.yaml)
- Multi-source training data with identity-offset unification
- Backbones via timm (resnet50, efficientnet_b0, vit_*_dinov2, ...)
- ArcFace / Triplet / ArcFace+Triplet losses
- Knowledge distillation mode (frozen teacher + cosine embedding loss)
- AMP, torch.compile, cosine LR schedule with warmup
- Early stopping on val Rank-1 (proxy validation built from training data)
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ============================================================================
# Config
# ============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ============================================================================
# Datasets
# ============================================================================

class MultiSourceReIDDataset(Dataset):
    """Concatenate multiple parquet sources, offsetting identity IDs per source
    so they occupy disjoint label ranges in the unified classifier."""

    def __init__(
        self,
        sources: List[dict],
        image_size: int = 224,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.split = split
        self.image_size = image_size

        frames = []
        for src in sources:
            root = src["root"]
            parquet = src["parquet"]
            offset = src.get("identity_offset", 0)

            df = pd.read_parquet(os.path.join(root, parquet))
            # Only keep rows whose image exists on disk
            df["_abs_path"] = df["image_path"].apply(lambda p: os.path.join(root, p))
            exists = df["_abs_path"].apply(os.path.exists)
            dropped = (~exists).sum()
            if dropped:
                print(f"  [{parquet}] dropping {dropped} missing images")
            df = df[exists].reset_index(drop=True)
            df["identity"] = df["identity"].astype(int) + int(offset)
            df["_source"] = root
            frames.append(df[["_abs_path", "identity", "_source"]])

        data = pd.concat(frames, ignore_index=True)

        if max_samples and len(data) > max_samples:
            data = data.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        # Relabel to dense 0..K-1 for ArcFace classifier
        unique_ids = sorted(data["identity"].unique())
        self.id_to_label = {pid: i for i, pid in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)
        self.labels = np.array([self.id_to_label[pid] for pid in data["identity"].values])
        self.original_ids = data["identity"].values
        self.paths = data["_abs_path"].tolist()
        self.sources = data["_source"].values

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), int(self.labels[idx])


def train_val_split(ds: MultiSourceReIDDataset, val_fraction: float, seed: int = 42):
    """Random split into train / val subsets, preserving the dataset's label map."""
    n = len(ds)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


class IndexSubset(Dataset):
    """Subset wrapper that also exposes labels array for sampler use."""
    def __init__(self, base: MultiSourceReIDDataset, indices: np.ndarray, transform=None):
        self.base = base
        self.indices = indices
        self.labels = base.labels[indices]
        self.transform = transform  # if None, use base.transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real = int(self.indices[i])
        if self.transform is None:
            return self.base[real]
        # Override transform (e.g. for val)
        image = Image.open(self.base.paths[real]).convert("RGB")
        return self.transform(image), int(self.base.labels[real])


# ============================================================================
# Losses
# ============================================================================

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        target_logits = torch.cos(theta + self.m * one_hot)
        logits = target_logits * self.s
        return F.cross_entropy(logits, labels)


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        e = F.normalize(embeddings, p=2, dim=1)
        dist = 1 - torch.mm(e, e.t())
        labels = labels.unsqueeze(0)
        same = labels == labels.t()

        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        for i in range(embeddings.size(0)):
            pos = same[i].clone(); pos[i] = False
            neg = ~same[i]
            if pos.any() and neg.any():
                hp = dist[i][pos].max()
                hn = dist[i][neg].min()
                loss = loss + F.relu(hp - hn + self.margin)
                count += 1
        return loss / max(count, 1)


def distill_loss(student_emb, teacher_emb):
    """1 - cosine similarity between L2-normalized embeddings."""
    s = F.normalize(student_emb, p=2, dim=1)
    t = F.normalize(teacher_emb, p=2, dim=1)
    return (1 - (s * t).sum(dim=1)).mean()


# ============================================================================
# Model
# ============================================================================

class ReIDModel(nn.Module):
    """Backbone (via timm) + projection head to fixed embedding dim."""
    def __init__(self, backbone_name: str, embedding_dim: int = 512, pretrained: bool = True,
                 backbone_kwargs: Optional[dict] = None):
        super().__init__()
        import timm
        kwargs = dict(backbone_kwargs or {})
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg", **kwargs
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.backbone_kwargs = kwargs

    def forward(self, x):
        return self.head(self.backbone(x))

    def encode(self, x):
        return F.normalize(self.forward(x), p=2, dim=1)


# ============================================================================
# Validation (proxy ReID on held-out subset of training data)
# ============================================================================

@torch.inference_mode()
def build_val_query_gallery(val_subset: IndexSubset):
    """For each identity with >=2 images in val, first is query, rest gallery."""
    labels = val_subset.labels
    by_id = {}
    for local_i, pid in enumerate(labels):
        by_id.setdefault(int(pid), []).append(local_i)

    query_local, gallery_local = [], []
    for pid, idxs in by_id.items():
        if len(idxs) >= 2:
            query_local.append(idxs[0])
            gallery_local.extend(idxs[1:])
    return query_local, gallery_local


@torch.inference_mode()
def encode_subset(model, subset: IndexSubset, indices: List[int], device, batch_size=128):
    model.eval()
    loader = DataLoader(
        IndexSubset(subset.base, subset.indices[indices], transform=subset.transform),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    embs, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        emb = model.encode(imgs)
        embs.append(emb.cpu().numpy())
        labels.append(lbls.numpy())
    return np.vstack(embs), np.concatenate(labels)


def compute_val_rank1(model, val_subset: IndexSubset, device) -> Tuple[float, float]:
    """Returns (rank1, mAP) on a proxy query/gallery built from val split."""
    q_local, g_local = build_val_query_gallery(val_subset)
    if not q_local or not g_local:
        return 0.0, 0.0

    q_emb, q_lbl = encode_subset(model, val_subset, q_local, device)
    g_emb, g_lbl = encode_subset(model, val_subset, g_local, device)

    sim = q_emb @ g_emb.T
    order = np.argsort(-sim, axis=1)
    g_ranked_lbl = g_lbl[order]
    matches = (g_ranked_lbl == q_lbl[:, None]).astype(np.int32)

    rank1 = matches[:, 0].mean()
    # Quick mAP
    aps = []
    for i in range(matches.shape[0]):
        rel = matches[i]
        if rel.sum() == 0:
            continue
        precision_at_k = rel.cumsum() / np.arange(1, len(rel) + 1)
        aps.append((precision_at_k * rel).sum() / rel.sum())
    mAP = float(np.mean(aps)) if aps else 0.0
    return float(rank1), mAP


# ============================================================================
# Training
# ============================================================================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def pick_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def maybe_compile(model, enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        try:
            return torch.compile(model)
        except Exception as e:
            print(f"  torch.compile failed ({e}); continuing uncompiled")
    return model


def build_teacher(cfg: dict, device: torch.device) -> Optional[nn.Module]:
    kd = cfg.get("distillation", {})
    if not kd.get("enabled", False):
        return None
    ckpt_path = kd["teacher_checkpoint"]
    teacher = ReIDModel(
        backbone_name=kd["teacher_backbone"],
        embedding_dim=kd.get("teacher_embedding_dim", 512),
        pretrained=False,
        backbone_kwargs=kd.get("teacher_backbone_kwargs"),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  Loaded teacher from {ckpt_path}")
    return teacher


def train(cfg: dict, args):
    set_seed(cfg.get("seed", 42))
    device = pick_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    exp_id = cfg["exp_id"]
    save_dir = Path(cfg.get("save_dir", "checkpoints")) / exp_id
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # Data
    full_ds = MultiSourceReIDDataset(
        sources=cfg["data"]["sources"],
        image_size=cfg["data"].get("image_size", 224),
        split="train",
        max_samples=cfg["data"].get("max_samples"),
        seed=cfg.get("seed", 42),
    )
    print(f"Total images: {len(full_ds)}, identities: {full_ds.num_classes}")

    train_idx, val_idx = train_val_split(full_ds, cfg["data"].get("val_fraction", 0.05))
    train_sub = IndexSubset(full_ds, train_idx)  # uses base train transform
    val_transform = transforms.Compose([
        transforms.Resize((cfg["data"].get("image_size", 224),) * 2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_sub = IndexSubset(full_ds, val_idx, transform=val_transform)
    print(f"Train: {len(train_sub)}  Val: {len(val_sub)}")

    loader = DataLoader(
        train_sub,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=cfg["train"].get("num_workers", 4) > 0,
    )

    # Model
    model = ReIDModel(
        backbone_name=cfg["backbone"]["name"],
        embedding_dim=cfg["head"]["embedding_dim"],
        pretrained=cfg["backbone"].get("pretrained", True),
        backbone_kwargs=cfg["backbone"].get("kwargs"),
    ).to(device)
    model = maybe_compile(model, cfg["train"].get("compile", False), device)

    # Losses
    loss_name = cfg["train"]["loss"]
    uses_arcface = "arcface" in loss_name
    uses_triplet = "triplet" in loss_name
    arc = ArcFaceLoss(
        cfg["head"]["embedding_dim"], full_ds.num_classes,
        s=cfg["train"].get("arcface_s", 30.0),
        m=cfg["train"].get("arcface_m", 0.5),
    ).to(device) if uses_arcface else None
    tri = TripletLoss(margin=cfg["train"].get("margin", 0.3)).to(device) if uses_triplet else None

    # Teacher
    teacher = build_teacher(cfg, device)

    # Optimizer
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if "backbone" in n: backbone_params.append(p)
        else: head_params.append(p)
    loss_params = list(arc.parameters()) if arc is not None else []
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["train"]["lr"] * 0.1},
        {"params": head_params,     "lr": cfg["train"]["lr"]},
        {"params": loss_params,     "lr": cfg["train"]["lr"]},
    ], weight_decay=cfg["train"].get("weight_decay", 1e-4))

    # Resume
    start_epoch, best_val = 0, -1.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", -1.0)
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, best_val {best_val:.4f})")

    # LR schedule (cosine with warmup)
    epochs = cfg["train"]["epochs"]
    total_steps = len(loader) * epochs
    warmup_steps = len(loader) * cfg["train"].get("warmup_epochs", 2)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch * len(loader) - 1
    )

    # AMP
    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    patience_used = 0
    patience = cfg.get("early_stop", {}).get("patience", 3)
    val_every = cfg.get("early_stop", {}).get("val_every_epochs", 1)

    kd_cfg = cfg.get("distillation", {})
    alpha = kd_cfg.get("alpha", 1.0)
    beta = kd_cfg.get("beta", 1.0)

    for epoch in range(start_epoch, epochs):
        model.train()
        if arc: arc.train()

        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                emb = model(images)
                loss_parts = {}
                total = 0.0
                if arc is not None:
                    la = arc(emb, labels)
                    loss_parts["arc"] = la.item()
                    total = total + la
                if tri is not None:
                    lt = tri(emb, labels)
                    loss_parts["tri"] = lt.item()
                    total = total + lt
                if teacher is not None:
                    with torch.no_grad():
                        t_emb = teacher(images)
                    ld = distill_loss(emb, t_emb)
                    loss_parts["kd"] = ld.item()
                    total = alpha * total + beta * ld

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += total.item()
            pbar.set_postfix(loss=f"{total.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}",
                             **{k: f"{v:.3f}" for k, v in loss_parts.items()})

        avg_loss = running / len(loader)

        # Validation
        val_rank1, val_map = 0.0, 0.0
        if (epoch + 1) % val_every == 0:
            val_rank1, val_map = compute_val_rank1(model, val_sub, device)
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}  val_rank1={val_rank1*100:.2f}%  val_mAP={val_map*100:.2f}%")

        history.append({"epoch": epoch+1, "loss": avg_loss,
                        "val_rank1": val_rank1, "val_mAP": val_map})
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

        backbone_kwargs = cfg["backbone"].get("kwargs") or {}
        # Always save latest
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val, "embedding_dim": cfg["head"]["embedding_dim"],
            "backbone_name": cfg["backbone"]["name"],
            "backbone_kwargs": backbone_kwargs,
        }, save_dir / "last.pth")

        # Best-by-val-rank1
        if val_rank1 > best_val:
            best_val = val_rank1; patience_used = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val, "val_rank1": val_rank1, "val_mAP": val_map,
                "embedding_dim": cfg["head"]["embedding_dim"],
                "backbone_name": cfg["backbone"]["name"],
                "backbone_kwargs": backbone_kwargs,
            }, save_dir / "best.pth")
            print(f"  Saved best (val_rank1={val_rank1*100:.2f}%)")
        elif (epoch + 1) % val_every == 0:
            patience_used += 1
            if patience_used >= patience:
                print(f"  Early stop: no val improvement for {patience} checks.")
                break

    print(f"\nTraining complete. Best val_rank1={best_val*100:.2f}%")
    print(f"Artifacts in: {save_dir}")


# ============================================================================
# Entry
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max_samples", type=int, default=None, help="override data.max_samples for debug")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs for debug")
    ap.add_argument("--device", default=None, help="override device (cuda/mps/cpu/auto)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_samples is not None:
        cfg.setdefault("data", {})["max_samples"] = args.max_samples
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    print(f"=== {cfg['exp_id']} ===")
    train(cfg, args)


if __name__ == "__main__":
    main()
