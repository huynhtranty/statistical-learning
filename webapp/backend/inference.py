"""Inference glue used by the FastAPI backend.

Loads the chosen model (default: YOLO — fastest of the three for the demo),
runs a single image through it, and returns COCO-style detection records.
"""
from __future__ import annotations

import os
from typing import Any

INPUT_SIZE = 640
DEFAULT_MODEL = os.environ.get("OD_MODEL", "yolo")           # yolo | faster_rcnn | detr
DEFAULT_WEIGHTS = os.environ.get("OD_WEIGHTS", "weights/yolo.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("OD_CONF", "0.25"))


def predict_image(image_bytes: bytes) -> dict[str, Any]:
    """Run inference on a single image (raw bytes) and return detections.

    Returns:
        {
          "model": str,
          "image_size": [width, height],
          "detections": [
            {"class": str, "class_id": int, "confidence": float, "bbox": [x, y, w, h]},
            ...
          ]
        }
    """
    # TODO: implement
    #   1. Decode bytes via PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB").
    #   2. Lazily load the chosen model + weights once (module-level cache).
    #   3. Run forward pass, filter by CONFIDENCE_THRESHOLD.
    #   4. Map class_id -> name via data/processed/annotations/classes.txt.
    #   5. Return JSON-serializable dict.
    raise NotImplementedError("predict_image: implement")
