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
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        raise RuntimeError(
            "pycocotools chưa sẵn sàng. Cài bằng `pip install pycocotools` rồi chạy lại."
        ) from exc

    gt = COCO(str(ground_truth_path))

    with predictions_path.open("r", encoding="utf-8") as f:
        predictions_data = json.load(f)
    if not isinstance(predictions_data, list):
        raise ValueError(f"Predictions file phải là list COCO results: {predictions_path}")

    model_size_mb = 0.0
    if weights_path is not None and weights_path.exists():
        model_size_mb = weights_path.stat().st_size / (1024.0 * 1024.0)

    if len(predictions_data) == 0:
        return {
            "mAP_50": 0.0,
            "mAP_50_95": 0.0,
            "per_class_AP": {},
            "fps": float(fps),
            "model_size_mb": float(model_size_mb),
        }

    dt = gt.loadRes(predictions_data)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    mAP_50_95 = float(ev.stats[0])
    mAP_50 = float(ev.stats[1])

    cat_ids = gt.getCatIds()
    cat_info = {cat["id"]: cat["name"] for cat in gt.loadCats(cat_ids)}
    precision = ev.eval.get("precision")
    # precision shape: [T, R, K, A, M]
    per_class_ap: dict[str, float] = {}
    if precision is not None:
        for k_idx, cat_id in enumerate(ev.params.catIds):
            p = precision[:, :, k_idx, 0, -1]
            p = p[p > -1]
            per_class_ap[cat_info.get(cat_id, str(cat_id))] = float(p.mean()) if p.size > 0 else 0.0

    return {
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "per_class_AP": per_class_ap,
        "fps": float(fps),
        "model_size_mb": float(model_size_mb),
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args.predictions, args.ground_truth, args.weights, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
