"""Train a YOLO object detector (Ultralytics implementation).

Conventions (shared across all three models in this project):
- Master annotation format: COCO JSON (converted to YOLO .txt for ultralytics)
- Train/val/test split: identical for all models (see scripts/split_dataset.py)
- Input resolution: 640 x 640
- Random seed: 42

Usage:
    python models/yolo/train.py \
        --data data/processed \
        --epochs 50 \
        --batch-size 16 \
        --output weights/yolo.pt
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
    p = argparse.ArgumentParser(description="Train YOLO via ultralytics.")
    p.add_argument("--data", type=Path, required=True, help="Processed data root.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", type=Path, required=True, help="Output checkpoint path (.pt).")
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
    #   1. Ensure YOLO-format labels exist (run scripts/convert_coco_to_yolo.py first).
    #   2. Build a temporary dataset YAML (path, train, val, names) for ultralytics.
    #   3. from ultralytics import YOLO; model = YOLO("yolov8n.pt"); model.train(...)
    #      with imgsz=INPUT_SIZE, seed=SEED, epochs/batch from args.
    #   4. Move best.pt to args.output.
    raise NotImplementedError("YOLO training: implement")


if __name__ == "__main__":
    main()
