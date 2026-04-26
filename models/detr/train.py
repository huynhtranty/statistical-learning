"""Train a DETR object detector (HuggingFace Transformers implementation).

Conventions (shared across all three models in this project):
- Master annotation format: COCO JSON
- Train/val/test split: identical for all models (see scripts/split_dataset.py)
- Input resolution: 640 x 640
- Random seed: 42

Usage:
    python models/detr/train.py \
        --data data/processed \
        --epochs 50 \
        --batch-size 4 \
        --output weights/detr.pth
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42
INPUT_SIZE = 640


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DETR via HuggingFace Transformers.")
    p.add_argument("--data", type=Path, required=True, help="Processed data root.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", type=Path, required=True, help="Output checkpoint path (.pth).")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    return p.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # TODO: implement training
    #   1. Load DetrImageProcessor + DetrForObjectDetection ("facebook/detr-resnet-50")
    #      with num_labels matching classes.txt and ignore_mismatched_sizes=True.
    #   2. Build a torch Dataset that yields (pixel_values, target) where target is COCO-style.
    #   3. Use HuggingFace Trainer or a manual loop with AdamW + LR scheduler per config.yaml.
    #   4. Save model.state_dict() to args.output.
    raise NotImplementedError("DETR training: implement")


if __name__ == "__main__":
    main()
