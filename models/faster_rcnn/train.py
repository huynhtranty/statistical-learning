"""Train a Faster R-CNN object detector (torchvision implementation).

Conventions (shared across all three models in this project):
- Master annotation format: COCO JSON
- Train/val/test split: identical for all models (see scripts/split_dataset.py)
- Input resolution: 640 x 640
- Random seed: 42

Usage:
    python models/faster_rcnn/train.py \
        --data data/processed \
        --epochs 50 \
        --batch-size 8 \
        --output weights/faster_rcnn.pth
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
    p = argparse.ArgumentParser(description="Train Faster R-CNN.")
    p.add_argument("--data", type=Path, required=True, help="Processed data root (contains images/ and annotations/).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
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
    #   1. Load COCO datasets (train.json, val.json) and DataLoaders with INPUT_SIZE=640.
    #   2. Build torchvision.models.detection.fasterrcnn_resnet50_fpn pretrained on COCO,
    #      replace the box predictor head for our num_classes (read from classes.txt).
    #   3. SGD optimizer + StepLR scheduler per config.yaml.
    #   4. Train loop with periodic validation; save best checkpoint to args.output.
    raise NotImplementedError("Faster R-CNN training: implement")


if __name__ == "__main__":
    main()
