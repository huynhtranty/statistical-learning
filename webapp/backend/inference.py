"""Inference glue used by the FastAPI backend.

Loads the chosen model (default: YOLO — fastest of the three for the demo),
runs a single image through it, and returns COCO-style detection records:

    {
      "model": str,
      "image_size": [width, height],
      "detections": [
        {"class": str, "class_id": int, "confidence": float, "bbox": [x, y, w, h]},
        ...
      ]
    }
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from PIL import Image

# Project root must be importable so we can reuse the trainer-branch model + decode code.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.test_and_visualize import load_model, run_predictions  # noqa: E402
from models.utils.coco_dataset import DEFAULT_CLASSES, get_class_names  # noqa: E402

INPUT_SIZE = int(os.environ.get("OD_INPUT_SIZE", "640"))
DEFAULT_MODEL = os.environ.get("OD_MODEL", "yolo")           # yolo | faster_rcnn | detr
DEFAULT_WEIGHTS = os.environ.get("OD_WEIGHTS", "weights/yolo.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("OD_CONF", "0.25"))
DEVICE = os.environ.get("OD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = int(os.environ.get("OD_NUM_CLASSES", "10"))

_model = None
_class_names: list[str] = []
_load_lock = Lock()


class ModelNotReady(RuntimeError):
    """Raised when the configured weights file is missing.

    The training team hasn't published a checkpoint yet; surface a clear
    message instead of a cryptic torch.load error.
    """


def _resolve_weights_path() -> Path:
    path = Path(DEFAULT_WEIGHTS)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _ensure_model_loaded() -> None:
    """Lazily load the model + class names once per process."""
    global _model, _class_names
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        weights_path = _resolve_weights_path()
        if not weights_path.exists():
            raise ModelNotReady(
                f"Weights not found at {weights_path}. "
                f"Set OD_WEIGHTS or wait for the trainer team to publish a checkpoint."
            )
        _model = load_model(DEFAULT_MODEL, str(weights_path), DEVICE, NUM_CLASSES)
        try:
            _class_names = get_class_names()
        except Exception:
            _class_names = list(DEFAULT_CLASSES)


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc


def predict_image(image_bytes: bytes) -> dict[str, Any]:
    pil_img = _decode_image(image_bytes)
    width, height = pil_img.size

    _ensure_model_loaded()

    # run_predictions takes a Path; spill bytes to a temp file so we don't
    # duplicate the (already fragile) decode logic from evaluation/test_and_visualize.py.
    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        pil_img.save(tmp, format="JPEG", quality=95)
        tmp_path = Path(tmp.name)
    try:
        boxes, labels, scores = run_predictions(
            _model,
            DEFAULT_MODEL,
            tmp_path,
            DEVICE,
            input_size=INPUT_SIZE,
            conf_threshold=CONFIDENCE_THRESHOLD,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    detections = []
    for box, class_id, score in zip(boxes, labels, scores):
        x, y, w, h = box
        cls_name = (
            _class_names[class_id] if 0 <= class_id < len(_class_names) else f"class_{class_id}"
        )
        detections.append(
            {
                "class": cls_name,
                "class_id": int(class_id),
                "confidence": float(score),
                "bbox": [float(x), float(y), float(w), float(h)],
            }
        )

    return {
        "model": DEFAULT_MODEL,
        "image_size": [width, height],
        "detections": detections,
    }
