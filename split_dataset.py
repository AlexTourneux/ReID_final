#!/usr/bin/env python3
"""
Split the on-disk test images into train/test parquets for model development.

The test.parquet that ships with the repo references all 30K images on disk
but labels everything "test". Since the original training images were not
distributed, we carve out our own split from what we have.

Split is done at the IDENTITY level (not image level) so that no individual
animal appears in both train and test — this is standard ReID protocol.

Output:
    datasets/dataset_a/our_train.parquet   (70% of identities)
    datasets/dataset_a/our_test.parquet    (30% of identities)

Usage:
    python3 split_dataset.py
    python3 split_dataset.py --train_ratio 0.7 --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd


def split(data_root: str, train_ratio: float, seed: int):
    rng = np.random.default_rng(seed)

    df = pd.read_parquet(os.path.join(data_root, "test.parquet"))

    # Only keep rows whose image file actually exists on disk
    df = df[df["image_path"].apply(
        lambda p: os.path.exists(os.path.join(data_root, p))
    )].reset_index(drop=True)
    print(f"Images confirmed on disk: {len(df)}")

    # Split identities — identities with only 1 image go to train since
    # they can't form a valid query/gallery pair in test
    identity_counts = df.groupby("identity").size()
    single_image_ids = identity_counts[identity_counts == 1].index.tolist()
    multi_image_ids  = identity_counts[identity_counts > 1].index.tolist()

    rng.shuffle(multi_image_ids)
    cutoff = int(len(multi_image_ids) * train_ratio)
    train_ids = set(multi_image_ids[:cutoff] + single_image_ids)
    test_ids  = set(multi_image_ids[cutoff:])

    train_df = df[df["identity"].isin(train_ids)].copy()
    test_df  = df[df["identity"].isin(test_ids)].copy()

    train_df["split"] = "train"
    test_df["split"]  = "test"

    out_train = os.path.join(data_root, "our_train.parquet")
    out_test  = os.path.join(data_root, "our_test.parquet")
    train_df.to_parquet(out_train, index=False)
    test_df.to_parquet(out_test,  index=False)

    print(f"\nSplit complete (seed={seed}, train_ratio={train_ratio})")
    print(f"  Train: {len(train_df):>6} images, {train_df['identity'].nunique():>5} identities → {out_train}")
    print(f"  Test:  {len(test_df):>6} images, {test_df['identity'].nunique():>5} identities → {out_test}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset_a into train/test parquets")
    parser.add_argument("--data_root",    type=str,   default="./datasets/dataset_a")
    parser.add_argument("--train_ratio",  type=float, default=0.70)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()
    split(args.data_root, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()
