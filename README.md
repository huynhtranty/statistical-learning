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
- **Reported metrics**: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-Score, FPS, Params, FLOPs, Confusion Matrix, PR Curve.

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
python models/faster_rcnn/train.py --data data --epochs 50 --batch_size 8 --output weights/faster_rcnn.pth

# YOLO
python models/yolo/train.py --data data --epochs 50 --batch_size 16 --output weights/yolo.pt

# DETR
python models/detr/train.py --data data --epochs 50 --batch_size 4 --output weights/detr.pth
```

Per-model details: [models/faster_rcnn/README.md](models/faster_rcnn/README.md), [models/yolo/README.md](models/yolo/README.md), [models/detr/README.md](models/detr/README.md).

## Evaluate

Shared evaluation pipeline reports identical metrics for every model:

```bash
# ─── Full Comprehensive Evaluation (RECOMMENDED) ─────────────────────────────
# Single model with all metrics: mAP, Precision, Recall, FPS, Params, FLOPs, Confusion Matrix, PR Curve
python evaluation/model_evaluation.py \
    --model faster_rcnn \
    --weights weights/faster_rcnn.pth \
    --predictions predictions/faster_rcnn_test.json \
    --ground-truth data/annotations/test.json \
    --num-classes 6 \
    --device cuda \
    --plot

# YOLO
python evaluation/model_evaluation.py \
    --model yolo \
    --weights weights/yolo.pt \
    --predictions predictions/yolo_test.json \
    --ground-truth data/annotations/test.json \
    --num-classes 6 \
    --device cuda \
    --plot

# DETR
python evaluation/model_evaluation.py \
    --model detr \
    --weights weights/detr.pth \
    --predictions predictions/detr_test.json \
    --ground-truth data/annotations/test.json \
    --num-classes 6 \
    --device cuda \
    --plot

# ─── Compare All Models ───────────────────────────────────────────────────────
# Run all three models and display comparison table
python evaluation/model_evaluation.py --compare-all --device cuda --plot

# ─── Speed Benchmark Only ─────────────────────────────────────────────────────
# Measure FPS and latency without detection metrics
python evaluation/benchmark_speed.py --weights weights/faster_rcnn.pth --model faster_rcnn --device cuda --iters 200
python evaluation/benchmark_speed.py --weights weights/yolo.pt --model yolo --device cuda --iters 200
python evaluation/benchmark_speed.py --weights weights/detr.pt --model detr --device cuda --iters 200

# ─── Basic mAP Evaluation (requires pycocotools) ─────────────────────────────
python evaluation/evaluate.py \
    --predictions predictions/yolo_test.json \
    --ground-truth data/processed/annotations/test.json \
    --weights weights/yolo.pt
```

### Output Metrics

| Metric | Description |
|--------|-------------|
| **mAP@0.5** | Mean Average Precision at IoU=0.5 |
| **mAP@0.5:0.95** | Mean AP averaged from IoU 0.5 to 0.95 |
| **Precision** | True Positives / (True Positives + False Positives) |
| **Recall** | True Positives / (True Positives + False Negatives) |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **FPS** | Frames per second (inference speed) |
| **Params** | Total trainable parameters (millions) |
| **FLOPs** | Floating point operations (billions) |
| **Confusion Matrix** | TP/FP/FN counts per class |
| **PR Curve** | Precision-Recall curve with AUC-PR |

### Results Location

- JSON results: `evaluation/results/evaluation.json`
- PR curve plots: `evaluation/results/<model>_pr_curve.png`

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
