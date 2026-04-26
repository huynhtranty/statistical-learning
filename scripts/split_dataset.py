"""Split a COCO-format dataset into train/val/test with a fixed seed.

Produces three COCO JSON files (train.json, val.json, test.json) so that all
three models consume the SAME split — required for fair comparison.

Default ratios: 70 / 15 / 15. Splitting is stratified per class to preserve
class balance across splits.

Usage:
    python scripts/split_dataset.py \
        --coco data/raw/annotations.json \
        --output data/processed/annotations \
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split a COCO dataset into train/val/test.")
    p.add_argument("--coco", type=Path, required=True, help="Source COCO JSON file.")
    p.add_argument("--output", type=Path, required=True, help="Output directory for split JSONs.")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=SEED, help="Random seed (fixed at 42 by default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1.0"
    args.output.mkdir(parents=True, exist_ok=True)
    # TODO: load COCO JSON, perform stratified split on image_id by dominant class,
    # write three COCO JSONs (train.json, val.json, test.json) preserving the
    # original `categories` list and the `info`/`licenses` blocks.
    raise NotImplementedError("split_dataset: implement stratified split")


if __name__ == "__main__":
    main()
