#!/usr/bin/env python3
"""Convert Market-1501 into parquets matching this project's schema.

Market-1501 layout (expected under --market_root):
    bounding_box_train/  -- 12,936 train images (751 identities)
    query/               --  3,368 query images (750 held-out ids)
    bounding_box_test/   -- 19,732 gallery images (same 750 + distractors)

Filename format: <pid>_c<cam>s<seq>_<frame>_<bbox>.jpg
    - pid = -1  → distractor (gallery only; keep as identity=-1)
    - pid =  0  → background / junk (gallery only; keep)
    - cam ∈ {1..6}

Output (written into --out_dir, default = market_root):
    market1501_train.parquet  columns = [image_path, identity, camera_id]
    market1501_test.parquet   columns = [image_path, identity, camera_id, split]

`image_path` is relative to --out_dir so train.py can join with the
source's `root`. Identities are offset by --id_offset (default 10000) so
they don't collide with dataset_a in a unified ArcFace classifier.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

FNAME_RE = re.compile(r"^(-?\d+)_c(\d+)s\d+_\d+_\d+\.jpg(?:\.jpg)?$")


def parse_split_dir(root: Path, subdir: str):
    dir_path = root / subdir
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Missing split directory: {dir_path}")
    rows = []
    for fname in sorted(os.listdir(dir_path)):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        pid = int(m.group(1))
        cam = int(m.group(2))
        rel = f"{subdir}/{fname}"
        rows.append({"image_path": rel, "identity": pid, "camera_id": cam})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market_root", required=True,
                    help="Path to Market-1501 directory containing bounding_box_train/, query/, bounding_box_test/")
    ap.add_argument("--out_dir", default=None,
                    help="Directory to write parquets (default: same as market_root)")
    ap.add_argument("--id_offset", type=int, default=10000,
                    help="Offset added to non-negative identities to avoid collision with dataset_a (default: 10000)")
    args = ap.parse_args()

    root = Path(args.market_root).resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else root
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading Market-1501 from: {root}")
    print(f"Writing parquets to:      {out}")

    train_rows = parse_split_dir(root, "bounding_box_train")
    query_rows = parse_split_dir(root, "query")
    gallery_rows = parse_split_dir(root, "bounding_box_test")

    def apply_offset(rows):
        for r in rows:
            # keep distractor (-1) / junk (0) ids negative/zero unshifted so ArcFace
            # can drop them; for train, -1 shouldn't appear, but guard anyway.
            if r["identity"] > 0:
                r["identity"] += args.id_offset

    apply_offset(train_rows)
    apply_offset(query_rows)
    apply_offset(gallery_rows)

    # Train: drop any pid <= 0 (shouldn't exist, but belt-and-braces)
    train_rows = [r for r in train_rows if r["identity"] > args.id_offset]

    # Rewrite image_path relative to out_dir (if out != root, use absolute-ish hint).
    # train.py joins root + image_path, so image_path must be relative to the src "root"
    # entry in configs. We standardize on `root = market_root`, paths relative to it.
    if out != root:
        print("NOTE: out_dir != market_root. In your YAML config set "
              f"`root: {root}` so image_path resolves correctly.")

    test_rows = []
    for r in query_rows:
        test_rows.append({**r, "split": "query"})
    for r in gallery_rows:
        test_rows.append({**r, "split": "gallery"})

    train_df = pd.DataFrame(train_rows, columns=["image_path", "identity", "camera_id"])
    test_df = pd.DataFrame(test_rows, columns=["image_path", "identity", "camera_id", "split"])

    train_path = out / "market1501_train.parquet"
    test_path = out / "market1501_test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print()
    print(f"market1501_train.parquet: {len(train_df)} rows, "
          f"{train_df['identity'].nunique()} identities")
    print(f"market1501_test.parquet:  {len(test_df)} rows "
          f"({(test_df['split']=='query').sum()} query / "
          f"{(test_df['split']=='gallery').sum()} gallery), "
          f"{test_df[test_df['identity']>0]['identity'].nunique()} identities")
    print()
    print("Saved:")
    print(f"  {train_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
