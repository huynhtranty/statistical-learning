"""Shared evaluation pipeline for all three models (Faster R-CNN, YOLO, DETR).

Reads predictions in COCO results format and computes the standard metrics
required by the project: mAP@0.5, mAP@0.5:0.95, per-class AP, FPS, model size.

Predictions JSON format (COCO results):
    [
      {"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float},
      ...
    ]

Usage:
    python evaluation/evaluate.py \
        --predictions runs/yolo/exp/predictions.json \
        --ground-truth data/processed/annotations/test.json \
        --weights weights/yolo.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict


class EvalResult(TypedDict):
    mAP_50: float
    mAP_50_95: float
    per_class_AP: dict[str, float]
    fps: float
    model_size_mb: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate detection predictions against COCO ground truth.")
    p.add_argument("--predictions", type=Path, required=True, help="COCO results JSON.")
    p.add_argument("--ground-truth", type=Path, required=True, help="COCO ground-truth JSON.")
    p.add_argument("--weights", type=Path, default=None, help="Optional checkpoint for model_size_mb.")
    p.add_argument("--fps", type=float, default=0.0, help="Optional FPS measurement (from benchmark_speed.py).")
    p.add_argument("--output", type=Path, default=Path("evaluation/results/metrics.json"))
    return p.parse_args()


def evaluate(
    predictions_path: Path,
    ground_truth_path: Path,
    weights_path: Path | None = None,
    fps: float = 0.0,
) -> EvalResult:
    """Compute mAP@0.5, mAP@0.5:0.95, per-class AP, FPS, model size.

    Returns a dict with keys: mAP_50, mAP_50_95, per_class_AP, fps, model_size_mb.
    """
    # TODO: implement using pycocotools
    #   from pycocotools.coco import COCO
    #   from pycocotools.cocoeval import COCOeval
    #   gt = COCO(str(ground_truth_path)); dt = gt.loadRes(str(predictions_path))
    #   ev = COCOeval(gt, dt, "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    #   - mAP_50_95 = ev.stats[0]
    #   - mAP_50    = ev.stats[1]
    #   - per_class_AP: iterate categories, slice ev.eval["precision"] and average
    raise NotImplementedError("evaluate: implement pycocotools-based mAP")


def main() -> None:
    args = parse_args()
    result = evaluate(args.predictions, args.ground_truth, args.weights, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
