#!/usr/bin/env python3
"""Ensemble multiple trained models by averaging L2-normalized embeddings.

Usage:
    python scripts/ensemble.py \\
        --checkpoints checkpoints/exp01/best.pth checkpoints/exp05/best.pth \\
        --weights 1.0 1.0 \\
        --dataset_root datasets/dataset_a \\
        --parquet our_test.parquet \\
        --protocol dataset_a \\
        --out predictions/ensemble_dataset_a.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.make_predictions import (  # noqa: E402
    PathsDataset,
    build_dataset_a_query_gallery,
    load_checkpoint,
)


@torch.inference_mode()
def encode(model, paths, device, batch_size, image_size, num_workers, image_size_override=None):
    size = image_size_override or image_size
    ds = PathsDataset(paths, image_size=size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    embs = []
    for imgs in tqdm(loader, desc=f"Encoding @ {size}"):
        imgs = imgs.to(device, non_blocking=True)
        embs.append(model.encode(imgs).cpu().numpy())
    return np.vstack(embs)


def l2norm(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="Per-model weight on its embedding contribution (default: equal)")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--protocol", choices=["dataset_a", "dataset_b"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    if args.weights is None:
        weights = [1.0] * len(args.checkpoints)
    else:
        if len(args.weights) != len(args.checkpoints):
            raise ValueError("--weights must match --checkpoints in length")
        weights = args.weights

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

    parquet_path = args.parquet if os.path.isabs(args.parquet) else os.path.join(args.dataset_root, args.parquet)
    df = pd.read_parquet(parquet_path)

    if args.protocol == "dataset_a":
        q_rows, g_rows, n_gallery_orig = build_dataset_a_query_gallery(df)
        query_paths = [os.path.join(args.dataset_root, r["image_path"]) for r in q_rows]
        gallery_paths = [os.path.join(args.dataset_root, r["image_path"]) for r in g_rows]
        self_match_offset = n_gallery_orig
    else:
        q_df = df[df["split"] == "query"].reset_index(drop=True)
        g_df = df[df["split"] == "gallery"].reset_index(drop=True)
        query_paths = [os.path.join(args.dataset_root, p) for p in q_df["image_path"].tolist()]
        gallery_paths = [os.path.join(args.dataset_root, p) for p in g_df["image_path"].tolist()]
        self_match_offset = None

    print(f"Queries: {len(query_paths)}  Gallery: {len(gallery_paths)}")

    q_acc, g_acc = None, None
    total_w = 0.0
    for ckpt, w in zip(args.checkpoints, weights):
        model, backbone, emb_dim = load_checkpoint(ckpt, device)
        print(f"  [{backbone}/{emb_dim}d, w={w}] {ckpt}")
        q_emb = l2norm(encode(model, query_paths, device, args.batch_size, args.image_size, args.num_workers))
        g_emb = l2norm(encode(model, gallery_paths, device, args.batch_size, args.image_size, args.num_workers))
        q_acc = (q_acc + w * q_emb) if q_acc is not None else (w * q_emb)
        g_acc = (g_acc + w * g_emb) if g_acc is not None else (w * g_emb)
        total_w += w
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    q_avg = l2norm(q_acc / total_w)
    g_avg = l2norm(g_acc / total_w)

    sim = q_avg @ g_avg.T
    if self_match_offset is not None:
        for i in range(len(query_paths)):
            sim[i, self_match_offset + i] = -np.inf

    top_k = min(args.top_k, sim.shape[1])
    order = np.argsort(-sim, axis=1)[:, :top_k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("query_index,ranked_gallery_indices\n")
        for i, row in enumerate(order):
            f.write(f'{i},"{",".join(map(str, row.tolist()))}"\n')
    print(f"Saved ensemble ({len(args.checkpoints)} models) to {out_path}")


if __name__ == "__main__":
    main()
