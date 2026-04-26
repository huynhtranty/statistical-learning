"""Convert COCO JSON annotations to YOLO format (one .txt file per image).

YOLO format per line: <class_id> <x_center> <y_center> <width> <height>
All coordinates normalized to [0, 1] relative to image size.

Usage:
    python scripts/convert_coco_to_yolo.py \
        --coco data/processed/annotations/train.json \
        --images data/processed/images/train \
        --output data/processed/annotations/yolo/train
"""
from __future__ import annotations

import argparse
from pathlib import Path

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert COCO JSON to YOLO .txt labels.")
    p.add_argument("--coco", type=Path, required=True, help="Path to COCO JSON file.")
    p.add_argument("--images", type=Path, required=True, help="Directory containing source images.")
    p.add_argument("--output", type=Path, required=True, help="Directory to write YOLO .txt files into.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # TODO: load COCO JSON, iterate annotations grouped by image_id,
    # convert (x, y, w, h) absolute -> (cx, cy, w, h) normalized, write per-image .txt files.
    raise NotImplementedError("convert_coco_to_yolo: implement conversion logic")


if __name__ == "__main__":
    main()
