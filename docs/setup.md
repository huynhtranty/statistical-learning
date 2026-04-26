# Setup

## Prerequisites

- **Python**: 3.10 or 3.11 (3.12 may have issues with some pinned wheels)
- **CUDA**: 11.8 or 12.1 if training on GPU (recommended)
- **Disk**: ~5 GB for dependencies, more for the dataset

## 1. Clone

```bash
git clone https://github.com/huynhtranty/statistical-learning.git
cd statistical-learning
```

## 2. Virtual environment

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you have a CUDA GPU, install the matching torch build first (visit https://pytorch.org/get-started/locally/) **before** running the command above to avoid a CPU-only install.

## 4. Download / prepare the dataset

> The dataset itself is not committed. Place raw downloads under `data/raw/`.

```bash
# After downloading raw files into data/raw/, run the split script:
python scripts/split_dataset.py \
    --coco data/raw/annotations.json \
    --output data/processed/annotations
```

For YOLO training, also generate YOLO-format labels:

```bash
python scripts/convert_coco_to_yolo.py \
    --coco data/processed/annotations/train.json \
    --images data/processed/images/train \
    --output data/processed/annotations/yolo/train
```

See [data/README.md](../data/README.md) for the dataset layout and class definitions.

## 5. Verify

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('ultralytics OK')"
python -c "import transformers; print('transformers', transformers.__version__)"
```

## 6. Train and evaluate

See the root [README.md](../README.md) for per-model training commands and evaluation.
