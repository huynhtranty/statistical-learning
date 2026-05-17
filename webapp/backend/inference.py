"""Inference glue used by the FastAPI backend.

Supports three trained models (Faster R-CNN, YOLO, DETR), lazily loaded into a
process-wide cache the first time each is requested. Returns COCO-style records:

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
from huggingface_hub import hf_hub_download

# Project root must be importable so we can reuse the trainer-branch model + decode code.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.test_and_visualize import load_model, run_predictions  # noqa: E402
from models.utils.coco_dataset import DEFAULT_CLASSES, get_class_names  # noqa: E402

INPUT_SIZE = int(os.environ.get("OD_INPUT_SIZE", "640"))
CONFIDENCE_THRESHOLD = float(os.environ.get("OD_CONF", "0.25"))
DEVICE = os.environ.get("OD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = int(os.environ.get("OD_NUM_CLASSES", "10"))

# Weights live in a separate HF Model repo so the Space stays under its 1 GB quota.
# Override with OD_WEIGHTS_REPO if you fork the project.
WEIGHTS_REPO = os.environ.get("OD_WEIGHTS_REPO", "hytaty/od-weights")

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "faster_rcnn": {"label": "Faster R-CNN", "filename": "faster_rcnn.pt"},
    "yolo": {"label": "YOLO", "filename": "yolo.pt"},
    "detr": {"label": "DETR", "filename": "detr.pt"},
}
DEFAULT_MODEL = os.environ.get("OD_MODEL", "faster_rcnn")

_model_cache: dict[str, Any] = {}
_class_names: list[str] = []
_load_lock = Lock()


class ModelNotReady(RuntimeError):
    """Raised when the requested model's weights file is missing on disk."""


class UnknownModel(ValueError):
    """Raised when the client asks for a model name not in the registry."""


def list_models() -> list[dict[str, Any]]:
    """Return registry entries — weights are downloaded lazily from HF Hub."""
    out = []
    for name, cfg in MODEL_REGISTRY.items():
        out.append(
            {
                "name": name,
                "label": cfg["label"],
                "available": True,
                "loaded": name in _model_cache,
                "default": name == DEFAULT_MODEL,
            }
        )
    return out


def _resolve_weights_path(model_name: str) -> Path:
    """Return a local path to the weights file, downloading from HF Hub if needed.

    A local `weights/<filename>` is preferred when present (handy for offline dev).
    Otherwise `hf_hub_download` fetches into the HF cache and returns the cached path.
    """
    cfg = MODEL_REGISTRY[model_name]
    local_override = PROJECT_ROOT / "weights" / cfg["filename"]
    if local_override.exists():
        return local_override
    cached = hf_hub_download(repo_id=WEIGHTS_REPO, filename=cfg["filename"])
    return Path(cached)


def _ensure_class_names() -> None:
    global _class_names
    if _class_names:
        return
    try:
        _class_names = get_class_names()
    except Exception:
        _class_names = list(DEFAULT_CLASSES)


def _ensure_model_loaded(model_name: str):
    """Lazily load `model_name` and keep it in the process-wide cache."""
    if model_name not in MODEL_REGISTRY:
        raise UnknownModel(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
        )
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached
    with _load_lock:
        cached = _model_cache.get(model_name)
        if cached is not None:
            return cached
        try:
            weights_path = _resolve_weights_path(model_name)
        except Exception as exc:
            raise ModelNotReady(
                f"Could not fetch weights for '{model_name}' from {WEIGHTS_REPO}: {exc}"
            ) from exc
        model = load_model(model_name, str(weights_path), DEVICE, NUM_CLASSES)
        _model_cache[model_name] = model
        _ensure_class_names()
        return model


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc


def predict_image(image_bytes: bytes, model_name: str | None = None) -> dict[str, Any]:
    chosen = model_name or DEFAULT_MODEL
    model = _ensure_model_loaded(chosen)

    pil_img = _decode_image(image_bytes)
    width, height = pil_img.size

    # run_predictions takes a Path; spill bytes to a temp file so we don't
    # duplicate the (already fragile) decode logic from evaluation/test_and_visualize.py.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_img.save(tmp, format="JPEG", quality=95)
        tmp_path = Path(tmp.name)
    try:
        boxes, labels, scores = run_predictions(
            model,
            chosen,
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
        "model": chosen,
        "image_size": [width, height],
        "detections": detections,
    }
