"""FastAPI backend for the object detection demo.

Endpoints:
- GET  /health   — liveness check.
- GET  /models   — registry of the three trained models + availability.
- POST /predict  — multipart form: `image` (required), `model` (optional name).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .inference import (
    ModelNotReady,
    UnknownModel,
    list_models,
    predict_image,
)

app = FastAPI(title="Object Detection Demo", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict[str, Any]:
    return {"models": list_models()}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model: str | None = Form(default=None),
) -> dict[str, Any]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    image_bytes = await image.read()
    try:
        return predict_image(image_bytes, model_name=model)
    except UnknownModel as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Serve the static frontend on the same origin so the single port works on
# Hugging Face Spaces (and locally). Mounted last so /health and /predict win.
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
