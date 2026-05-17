# Web App

A minimal demo: upload an image, see detected objects drawn on top.

- **Backend**: FastAPI (`webapp/backend/main.py`) — `GET /health` + `POST /predict`
- **Frontend**: a single static HTML file (`webapp/frontend/index.html`) with a canvas overlay — no build step

The webapp reuses model loading and prediction helpers from
[`evaluation/test_and_visualize.py`](../evaluation/test_and_visualize.py) so the
inference path stays in sync with the rest of the project.

## Run

Both commands must be launched from the **project root**.

```bash
# 1. Make sure weights exist (default: weights/yolo.pt).
#    Until the trainer team publishes a checkpoint, /predict returns 503.
ls weights/

# 2. Start the API
uvicorn webapp.backend.main:app --reload --host 0.0.0.0 --port 8000

# 3. Open the frontend
python -m http.server 5173 --directory webapp/frontend
# then visit http://localhost:5173
```

## Choosing the model

Override via env vars before starting uvicorn:

```bash
OD_MODEL=faster_rcnn OD_WEIGHTS=weights/faster_rcnn.pt \
    uvicorn webapp.backend.main:app --reload
```

| Variable        | Default            | Notes                                       |
| --------------- | ------------------ | ------------------------------------------- |
| `OD_MODEL`      | `yolo`             | `yolo` \| `faster_rcnn` \| `detr`           |
| `OD_WEIGHTS`    | `weights/yolo.pt`  | Path resolved relative to the project root  |
| `OD_CONF`       | `0.25`             | Min confidence for returned detections      |
| `OD_INPUT_SIZE` | `640`              | Must match the training preprocessing       |
| `OD_NUM_CLASSES`| `10`               | Match the trained YOLO head                 |
| `OD_DEVICE`     | auto               | `cuda` if available, else `cpu`             |

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
    {"class": "cat", "class_id": 0, "confidence": 0.91, "bbox": [120, 80, 200, 400]},
    {"class": "dog", "class_id": 1, "confidence": 0.87, "bbox": [600, 300, 250, 180]}
  ]
}
```

`bbox` is `[x, y, w, h]` in absolute pixel coordinates of the original image.

Status codes:
- `400` — request body was not a valid image
- `503` — `OD_WEIGHTS` not found (training not finished, or path misconfigured)
