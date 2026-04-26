# Web App

A minimal demo: upload an image, see detected objects.

- **Backend**: FastAPI (`webapp/backend/main.py`) — single `POST /predict` endpoint
- **Frontend**: a single static HTML file (`webapp/frontend/index.html`) — no build step

## Run

```bash
# 1. Make sure weights exist (default: weights/yolo.pt)
ls weights/

# 2. Start the API
uvicorn webapp.backend.main:app --reload --host 0.0.0.0 --port 8000

# 3. Open the frontend
#    Either open webapp/frontend/index.html directly in a browser,
#    or serve it with any static server, e.g.:
python -m http.server 5173 --directory webapp/frontend
# then visit http://localhost:5173
```

## Choosing the model

Override via env vars before starting uvicorn:

```bash
OD_MODEL=faster_rcnn OD_WEIGHTS=weights/faster_rcnn.pth \
    uvicorn webapp.backend.main:app --reload
```

Supported `OD_MODEL` values: `yolo`, `faster_rcnn`, `detr`. Default: `yolo` (fastest for the demo).

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /predict`

`multipart/form-data` with field `image` (any common image format).

Response:

```json
{
  "model": "yolo",
  "image_size": [1280, 720],
  "detections": [
    {"class": "person", "class_id": 0, "confidence": 0.91, "bbox": [120, 80, 200, 400]},
    {"class": "car",    "class_id": 1, "confidence": 0.87, "bbox": [600, 300, 250, 180]}
  ]
}
```

`bbox` is `[x, y, w, h]` in absolute pixel coordinates.
