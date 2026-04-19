#!/usr/bin/env python3
"""Generate a ranked-gallery CSV from a trained checkpoint.

Supports:
  - dataset_a-style parquets (single parquet, dynamic query/gallery via the
    `load_dataset_a_gt` protocol used in evaluate.py: for each identity with
    >=2 images, first 2 become queries, rest gallery, and queries are also
    appended to gallery). Self-matches are suppressed.
  - dataset_b-style parquets (single parquet with a `split` column holding
    "query"/"gallery").

Output CSV matches the format expected by evaluate.py:
    query_index,ranked_gallery_indices
    0,"45,12,78,..."

Usage:
    python scripts/make_predictions.py \\
        --checkpoint checkpoints/exp01/best.pth \\
        --dataset_root datasets/dataset_a \\
        --parquet our_test.parquet \\
        --protocol dataset_a \\
        --out predictions/exp01_dataset_a.csv

    python scripts/make_predictions.py \\
        --checkpoint checkpoints/exp01/best.pth \\
        --dataset_root path/to/market1501 \\
        --parquet market1501_test.parquet \\
        --protocol dataset_b \\
        --out predictions/exp01_market.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Make the project root importable so we can reuse ReIDModel from train.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import ReIDModel  # noqa: E402


class PathsDataset(Dataset):
    def __init__(self, paths, image_size=224):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    backbone = ckpt.get("backbone_name")
    emb_dim = ckpt.get("embedding_dim", 512)
    backbone_kwargs = ckpt.get("backbone_kwargs") or {}
    if backbone is None:
        raise ValueError(
            f"Checkpoint at {path} has no 'backbone_name'. "
            "Pass --backbone and --embedding_dim to override.")
    model = ReIDModel(backbone_name=backbone, embedding_dim=emb_dim,
                      pretrained=False, backbone_kwargs=backbone_kwargs).to(device)
    # Strip torch.compile prefix if present
    state = ckpt["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, backbone, emb_dim


@torch.inference_mode()
def encode_paths(model, paths, device, batch_size=128, image_size=224, num_workers=4):
    ds = PathsDataset(paths, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    embs = []
    for imgs in tqdm(loader, desc="Encoding"):
        imgs = imgs.to(device, non_blocking=True)
        emb = model.encode(imgs)
        embs.append(emb.cpu().numpy())
    return np.vstack(embs)


def build_dataset_a_query_gallery(df):
    """Replicates evaluate.py's load_dataset_a_gt split.

    For each identity:
      - if >=2 images: first 2 rows → queries, rest → gallery
      - else: all rows → gallery
    Then queries are appended to gallery (standard ReID protocol).

    Returns (query_paths, gallery_paths, query_rel, gallery_rel, n_q) where rel
    are parquet-relative paths for bookkeeping. Gallery self-match indices
    (query i → gallery[n_gallery_orig + i]) are returned so we can mask them.
    """
    query_rows, gallery_rows = [], []
    for pid, group in df.groupby("identity", sort=False):
        rows = group.to_dict("records")
        if len(rows) >= 2:
            query_rows.extend(rows[:2])
            gallery_rows.extend(rows[2:])
        else:
            gallery_rows.extend(rows)
    n_gallery_orig = len(gallery_rows)
    gallery_rows.extend(query_rows)
    return query_rows, gallery_rows, n_gallery_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_root", required=True,
                    help="Root directory; image_path column is joined relative to this")
    ap.add_argument("--parquet", required=True,
                    help="Parquet filename (inside --dataset_root) or absolute path")
    ap.add_argument("--protocol", choices=["dataset_a", "dataset_b"], required=True)
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model, backbone, emb_dim = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {backbone} ({emb_dim}-d) from {args.checkpoint}")

    parquet_path = args.parquet if os.path.isabs(args.parquet) else os.path.join(args.dataset_root, args.parquet)
    df = pd.read_parquet(parquet_path)

    # Build query / gallery path lists per protocol
    if args.protocol == "dataset_a":
        query_rows, gallery_rows, n_gallery_orig = build_dataset_a_query_gallery(df)
        query_paths = [os.path.join(args.dataset_root, r["image_path"]) for r in query_rows]
        gallery_paths = [os.path.join(args.dataset_root, r["image_path"]) for r in gallery_rows]
        self_match_offset = n_gallery_orig  # query i sits at gallery[n_gallery_orig + i]
    else:
        q_df = df[df["split"] == "query"].reset_index(drop=True)
        g_df = df[df["split"] == "gallery"].reset_index(drop=True)
        query_paths = [os.path.join(args.dataset_root, p) for p in q_df["image_path"].tolist()]
        gallery_paths = [os.path.join(args.dataset_root, p) for p in g_df["image_path"].tolist()]
        self_match_offset = None

    # Verify files exist
    missing_q = sum(1 for p in query_paths if not os.path.exists(p))
    missing_g = sum(1 for p in gallery_paths if not os.path.exists(p))
    if missing_q or missing_g:
        print(f"WARNING: {missing_q}/{len(query_paths)} query and "
              f"{missing_g}/{len(gallery_paths)} gallery images missing on disk")

    print(f"Queries: {len(query_paths)}  Gallery: {len(gallery_paths)}")

    q_emb = encode_paths(model, query_paths, device,
                         batch_size=args.batch_size, image_size=args.image_size,
                         num_workers=args.num_workers)
    g_emb = encode_paths(model, gallery_paths, device,
                         batch_size=args.batch_size, image_size=args.image_size,
                         num_workers=args.num_workers)

    # Cosine similarity (already L2-normalized via .encode)
    sim = q_emb @ g_emb.T  # (n_q, n_g)

    # Suppress self-matches for dataset_a protocol (query i is at gallery[n_orig + i])
    if self_match_offset is not None:
        for i in range(len(query_paths)):
            sim[i, self_match_offset + i] = -np.inf

    top_k = min(args.top_k, sim.shape[1])
    # Partial sort for speed on very wide gallery
    order = np.argsort(-sim, axis=1)[:, :top_k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("query_index,ranked_gallery_indices\n")
        for i, row in enumerate(order):
            f.write(f'{i},"{",".join(map(str, row.tolist()))}"\n')
    print(f"Saved {len(query_paths)} rankings to {out_path}")


if __name__ == "__main__":
    main()
