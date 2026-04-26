"""Convert COCO JSON annotations to Pascal VOC XML format (one .xml per image).

Usage:
    python scripts/convert_coco_to_voc.py \
        --coco data/processed/annotations/train.json \
        --images data/processed/images/train \
        --output data/processed/annotations/voc/train
"""
from __future__ import annotations

import argparse
from pathlib import Path

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert COCO JSON to Pascal VOC XML.")
    p.add_argument("--coco", type=Path, required=True, help="Path to COCO JSON file.")
    p.add_argument("--images", type=Path, required=True, help="Directory containing source images.")
    p.add_argument("--output", type=Path, required=True, help="Directory to write VOC .xml files into.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # TODO: load COCO JSON, build a Pascal VOC <annotation> XML tree per image with
    # <object><name>...</name><bndbox><xmin/><ymin/><xmax/><ymax/></bndbox></object>.
    raise NotImplementedError("convert_coco_to_voc: implement conversion logic")


if __name__ == "__main__":
    main()
