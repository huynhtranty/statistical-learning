"""FastAPI backend exposing a single /predict endpoint for the object detection demo.

POST /predict (multipart/form-data) with field `image` returns:
    {
      "model": "yolo",
      "image_size": [width, height],
      "detections": [
        {"class": "person", "class_id": 0, "confidence": 0.91, "bbox": [x, y, w, h]},
        ...
      ]
    }
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import predict_image

app = FastAPI(title="Object Detection Demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    image_bytes = await image.read()
    return predict_image(image_bytes)
