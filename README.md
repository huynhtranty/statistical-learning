# statistical-learning

---

# Object Detection Final Project

A comparative study and deployable web application for object detection, benchmarking three representative architectures: **Faster R-CNN** (two-stage), **YOLO** (one-stage), and **DETR** (transformer-based).

## Project Goal

Train and evaluate three object detection models on the same dataset, with the same train/val/test split, the same input resolution, and the same metrics — then expose the best model through a small FastAPI web application that accepts an uploaded image and returns predicted bounding boxes.

## Team and Responsibility Split

3 members: 2 Computer Science (CS) + 1 Software Engineering (SE).

| Member | Role | Responsibility |
|--------|------|----------------|
| CS #1  | ML Engineer    | Faster R-CNN training pipeline (`models/faster_rcnn/`), data preprocessing scripts (`scripts/`), shared evaluation pipeline (`evaluation/evaluate.py`) |
| CS #2  | ML Engineer    | YOLO and DETR training pipelines (`models/yolo/`, `models/detr/`), speed benchmarking (`evaluation/benchmark_speed.py`), per-class analysis |
| SE     | Software Eng.  | FastAPI backend + inference glue (`webapp/backend/`), frontend upload form (`webapp/frontend/`), packaging, deployment, documentation |

## Tech Stack

- **Deep learning**: [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/) (Faster R-CNN)
- **YOLO**: [Ultralytics YOLO](https://docs.ultralytics.com/)
- **DETR**: [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- **Evaluation**: [pycocotools](https://github.com/cocodataset/cocoapi) for mAP
- **Web app**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Image I/O**: OpenCV, Pillow

## Conventions (enforced across all models)

- **Master annotation format**: COCO JSON. All converters target this as the source of truth.
- **Identical train/val/test split** across all three models (see `scripts/split_dataset.py`).
- **Standardized input resolution**: 640 x 640.
- **Random seed**: 42 in every training script.
- **Reported metrics**: mAP@0.5, mAP@0.5:0.95, per-class AP, FPS, model size (MB), training time.

## Setup

See [docs/setup.md](docs/setup.md) for full instructions. Short version:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Each model has its own training entrypoint and config. All accept the same CLI flags.

```bash
# Faster R-CNN
python models/faster_rcnn/train.py --data data/processed --epochs 50 --batch-size 8 --output weights/faster_rcnn.pth

# YOLO
python models/yolo/train.py --data data/processed --epochs 50 --batch-size 16 --output weights/yolo.pt

# DETR
python models/detr/train.py --data data/processed --epochs 50 --batch-size 4 --output weights/detr.pth
```

Per-model details: [models/faster_rcnn/README.md](models/faster_rcnn/README.md), [models/yolo/README.md](models/yolo/README.md), [models/detr/README.md](models/detr/README.md).

## Evaluate

Shared evaluation pipeline reports identical metrics for every model:

```bash
python evaluation/evaluate.py --predictions <preds.json> --ground-truth data/processed/annotations/test.json
python evaluation/benchmark_speed.py --weights weights/yolo.pt --model yolo
```

## Run the Web App

```bash
uvicorn webapp.backend.main:app --reload --host 0.0.0.0 --port 8000
# then open webapp/frontend/index.html in a browser
```

See [webapp/README.md](webapp/README.md) for API details.

## Repository Layout

```
.
├── data/          # Raw + processed datasets (gitignored content)
├── scripts/       # Data conversion + split utilities
├── models/        # Training scripts and configs per model
├── evaluation/    # Shared metrics + speed benchmarks
├── webapp/        # FastAPI backend + simple frontend
├── weights/       # Trained checkpoints (gitignored)
├── notebooks/     # Exploration
├── report/        # Final write-up
└── docs/          # Setup + architecture notes
```
