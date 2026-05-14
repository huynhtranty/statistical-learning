# YOLO (You Only Look Once)

## Mô tả

YOLO là mô hình **one-stage detector** nổi tiếng với tốc độ inference nhanh.
Implementation trong project là phiên bản đơn giản hoá tự xây, gồm 3 phần:

1. **Backbone** — mặc định **ResNet-34 ImageNet pretrained** (torchvision); option `use_csp_backbone=True` để dùng CSPDarkNet from-scratch cho phần "tự xây" trong báo cáo.
2. **Neck** (FPN-style) — top-down pathway kết hợp feature ở 3 scale (stride 8/16/32). Channels (128, 256, 512) khớp ResNet-34.
3. **Head** — predict (tx, ty, tw, th, objectness, num_classes) cho 3 anchor / cell ở mỗi scale.

## Loss — chi tiết quan trọng

`YOLOLoss` trong [models/utils/losses.py](../utils/losses.py) đã được refactor theo phong cách YOLOv5:

- **Center-region assignment**: mỗi GT được gán cho **1-3 cells** (center + neighbor cùng nửa cell), không phải chỉ 1 cell. Tăng positive samples 1.5-3x → gradient mạnh hơn → hội tụ nhanh hơn trên dataset nhỏ.
- **YOLOv5 xy parameterization**: `pred_xy = 2·sigmoid(t) − 0.5` ∈ [-0.5, 1.5] — cho phép neighbor cells dự đoán đúng center của GT.
- **YOLOv5 wh parameterization**: `pred_wh = (2·sigmoid(t))²` ∈ [0, 4] — gradient ổn định ở large box, không saturate như sigmoid thuần.
- **Box loss**: GIoU + L1 trên (cx, cy, w, h) chuẩn hoá. L1 giúp khi GIoU saturate (pred lọt hoàn toàn trong GT).
- **Focal objectness**: α=0.25, γ=2.0 — chống class imbalance pos/neg trên grid.
- **Scale assignment**: max-side < 0.15 → P3 (small grid), < 0.45 → P4, else → P5. Dùng max-side thay area để chống lệch khi box dạng "thanh dài".

## Head bias init (prior matching dataset)

[head.py](head.py) init bias để output ban đầu khớp data statistics:

| Slot | Bias | Decoded value | Ý nghĩa |
|---|---|---|---|
| `tx, ty` | 0 | `2·σ(0) − 0.5 = 0.5` | center của cell |
| `tw, th` | -1.18 | `(2·σ(-1.18))² ≈ 0.22` | mean object size ~22% ảnh |
| `obj` | -4.6 | `σ(-4.6) = 0.01` | negative prior |
| `cls` | -log((1-p)/p) với p=1/num_classes | `1/num_classes` | uniform |

Với init này, model có thể detect được object ở pretrained backbone mà không cần ép training quá nhiều.

## Cấu trúc file

| File | Mô tả |
|------|-------|
| `model.py` | `build_yolo()` — ghép backbone + neck + head, tự normalize ImageNet bên trong |
| `backbone.py` | `ResNet34Backbone` (ImageNet pretrained) + `YOLOBackbone` (CSP from-scratch) |
| `neck.py` | `YOLONeck` — FPN top-down |
| `head.py` | `SingleScaleHead` + `YOLODetectionHead` — output 3 anchor × (5+C) channels |
| `train.py` | Training loop với YOLOLoss |
| `inference.py` | Inference script với NMS |
| `config.yaml` | Cấu hình model, training, và inference (conf/iou thresholds) |

## Cách chạy training

```bash
# Train (default: pretrained ResNet-34 backbone + augmentation tắt)
python models/yolo/train.py --data_root data --epochs 30 --batch_size 8 \
  --output weights/yolo.pt --device cuda

# Train kèm augmentation (horizontal flip + color jitter)
python models/yolo/train.py --data_root data --epochs 30 --batch_size 8 \
  --augment --output weights/yolo.pt --device cuda

# Resume
python models/yolo/train.py --resume weights/yolo.pt --epochs 20 --device cuda
```

Checkpoints lưu ở `models/yolo/checkpoints/best_model.pt`, logs tensorboard ở `models/yolo/logs/`.

## Cách inference / visualize

```bash
python evaluation/test_and_visualize.py --model yolo \
  --weights weights/yolo.pt \
  --data data/images/test --output evaluation/results/yolo_vis \
  --device cuda --conf-threshold 0.25 \
  --show-gt --ann-file data/annotations/test.json
```

> `--conf-threshold 0.25` là default hợp lý. 0.05 sẽ cho rất nhiều false positives. 0.5 chỉ hợp lý khi model đã fine-tune kỹ.

## Debug khi kết quả tệ

Đọc [README chính phần "Debug / Sanity Check"](../../README.md#debug--sanity-check-yolo). Thứ tự:

```bash
# 1) Vẽ GT từ dataloader để verify dataset
python scripts/visualize_dataset.py --data_root data --split train --num 12 --augment

# 2) Inspect 1 batch
python scripts/debug_one_batch.py --data_root data --device cuda

# 3) Overfit 1 batch — pipeline phải học được đến loss < 0.5
python scripts/overfit_one_batch.py --data_root data --steps 500 --lr 1e-3 --device cuda
```

Nếu overfit không xuống < 1.0 sau 500 steps → bug ở loss/model. Nếu overfit OK nhưng full train flat → undertrain hoặc data quá ít/noisy.

## Convention dữ liệu

- Dataset trả `boxes` ở **xyxy pixel trong ảnh đã resize 640×640**.
- YOLOLoss tự normalize `/640` để có cxcywh ∈ [0,1].
- Inference decode trả về **xywh trong ảnh GỐC** (sau khi divide scale_x/scale_y).
