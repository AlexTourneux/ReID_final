#!/usr/bin/env python3
"""Benchmark a trained checkpoint for efficiency reporting.

Measures:
  - parameter count (total, trainable)
  - latency at batch=1 (ms/image) averaged over N timed iterations after warmup
  - throughput at batch=64 (images/sec) — more realistic for retrieval pipelines
  - peak GPU memory (MB) during a batch=64 forward pass

Usage:
    python scripts/benchmark.py --checkpoint checkpoints/exp01/best.pth
    python scripts/benchmark.py --checkpoint checkpoints/exp05/best.pth --batch_size 128
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import ReIDModel  # noqa: E402


def load_checkpoint(path, device, backbone_override=None, emb_dim_override=None):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    backbone = backbone_override or ckpt.get("backbone_name")
    if backbone is None:
        raise ValueError(
            f"Checkpoint has no backbone_name key (old format). "
            "Pass --backbone resnet50 to specify it manually.")
    emb_dim = emb_dim_override or ckpt.get("embedding_dim", 512)
    backbone_kwargs = ckpt.get("backbone_kwargs") or {}
    model = ReIDModel(backbone_name=backbone, embedding_dim=emb_dim,
                      pretrained=False, backbone_kwargs=backbone_kwargs).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, backbone, emb_dim


@torch.inference_mode()
def time_forward(model, device, batch_size, image_size, warmup, iters):
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    per_batch_ms = (elapsed / iters) * 1000.0
    imgs_per_sec = (batch_size * iters) / elapsed
    return per_batch_ms, imgs_per_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone", default=None,
                    help="Override backbone name (needed for old-format checkpoints "
                    "that lack a backbone_name key, e.g. --backbone resnet50)")
    ap.add_argument("--embedding_dim", type=int, default=None,
                    help="Override embedding dim (old-format checkpoints only)")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for throughput measurement")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    else:
        device = torch.device(args.device)

    model, backbone, emb_dim = load_checkpoint(
        args.checkpoint, device, args.backbone, args.embedding_dim)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Backbone:      {backbone}")
    print(f"Embedding dim: {emb_dim}")
    print(f"Device:        {device}")
    print(f"Params:        {total_params/1e6:.2f}M total ({trainable_params/1e6:.2f}M trainable)")
    print()

    # Latency at batch=1
    lat_ms, _ = time_forward(model, device, 1, args.image_size, args.warmup, args.iters)
    print(f"Latency @ bs=1:    {lat_ms:.2f} ms/image  ({1000/lat_ms:.1f} img/s)")

    # Throughput at args.batch_size
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _, thr = time_forward(model, device, args.batch_size, args.image_size, args.warmup, args.iters)
    print(f"Throughput @ bs={args.batch_size}: {thr:.1f} img/s")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"Peak GPU memory:   {peak_mb:.1f} MB  (high-water mark, CUDA)")
    elif device.type == "mps":
        mem_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
        print(f"GPU memory:        {mem_mb:.1f} MB  (MPS snapshot — not peak high-water mark)")
    else:
        print("Peak memory:       (GPU only — run on CUDA or MPS)")

    print()
    print("RESULTS.md cells:")
    print(f"  throughput: {thr:.0f} img/s")
    if device.type == "cuda":
        print(f"  peak_mem:   {peak_mb:.0f} MB  (CUDA peak)")
    elif device.type == "mps":
        print(f"  peak_mem:   {mem_mb:.0f} MB  (MPS snapshot, not true peak)")
    print(f"  params:     {total_params/1e6:.1f}M")


if __name__ == "__main__":
    main()
