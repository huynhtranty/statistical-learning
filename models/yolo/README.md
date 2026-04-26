# YOLO

One-stage detector via [Ultralytics YOLOv8](https://docs.ultralytics.com/). Selected as the **one-stage / real-time** representative — designed for high FPS and a strong accuracy/latency tradeoff.

## Model variant

`yolov8n` (nano) by default — light enough for CPU inference in the web app demo. Switch to `yolov8s` / `yolov8m` in `config.yaml` for higher mAP at the cost of FPS.

## Training

```bash
# 1. Convert COCO JSON to YOLO .txt labels (one-time)
python scripts/convert_coco_to_yolo.py \
    --coco data/processed/annotations/train.json \
    --images data/processed/images/train \
    --output data/processed/annotations/yolo/train

# 2. Train
python models/yolo/train.py \
    --data data/processed \
    --epochs 50 \
    --batch-size 16 \
    --output weights/yolo.pt
```

## Hyperparameters (see `config.yaml`)

| Param         | Value      |
|---------------|------------|
| Input size    | 640 x 640  |
| Optimizer     | SGD (lr0 0.01, momentum 0.937, wd 5e-4) |
| Epochs        | 50         |
| Batch size    | 16         |
| Mosaic aug    | enabled    |
| Random seed   | 42         |

## Expected output

- Best checkpoint: `weights/yolo.pt`
- Ultralytics logs and curves under `runs/yolo/exp/`
- Final test-set evaluation runs through `evaluation/evaluate.py` for COCO mAP
