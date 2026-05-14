# YOLO (You Only Look Once)

## Mô tả

YOLO là mô hình **one-stage detector** nổi tiếng với tốc độ inference nhanh.
Kiến trúc trong project này là phiên bản đơn giản hoá tự xây dựng, gồm 3 phần:

1. **Backbone** — mặc định dùng **ResNet-34 ImageNet pretrained** (torchvision) để feature extractor mạnh ngay từ đầu; có thể chuyển về CSPDarkNet from-scratch nếu muốn.
2. **Neck** (FPN-style): top-down pathway kết hợp feature map ở 3 scale (stride 8/16/32).
3. **Detection Head**: dự đoán bounding box (tx, ty, tw, th), objectness score, và class logits cho 3 anchor / cell.

## Cấu trúc file

| File | Mô tả |
|------|-------|
| `model.py` | Hàm `build_yolo()` — ghép backbone + neck + head |
| `backbone.py` | `YOLOBackbone` (CSP from-scratch) + `ResNet34Backbone` (ImageNet pretrained) |
| `neck.py` | `YOLONeck` — FPN top-down, in_channels (128, 256, 512) khớp ResNet-34 |
| `head.py` | `YOLODetectionHead` — dự đoán detection ở mỗi scale, bias init prior (obj≈0.01, cls≈1/num_classes, wh≈0.1) |
| `train.py` | Training loop hoàn chỉnh với YOLOLoss (focal obj + GIoU box) |
| `config.yaml` | Cấu hình model (num_classes, image_size, anchors) |

## Cách chạy training

```bash
# Train với pretrained ResNet-34 backbone (mặc định)
python models/yolo/train.py --data_root data --epochs 10 --batch_size 8 \
  --output weights/yolo.pt --device cuda

# Train với augmentation
python models/yolo/train.py --data_root data --epochs 30 --batch_size 8 \
  --augment --output weights/yolo.pt --device cuda

# Resume từ checkpoint
python models/yolo/train.py --resume weights/yolo.pt --epochs 20 --device cuda
```

Checkpoints lưu ở `models/yolo/checkpoints/best_model.pt`, logs tensorboard ở `models/yolo/logs/`.

## Cách inference / visualize

```bash
# Vẽ predictions + GT
python evaluation/test_and_visualize.py --model yolo \
  --weights weights/yolo.pt \
  --data data/images/test --output evaluation/results/yolo_vis \
  --device cuda --conf-threshold 0.25 \
  --show-gt --ann-file data/annotations/test.json
```

## Loss & Encode/Decode

- **Box param**: `tx, ty ∈ (0,1)` = offset trong cell; `tw, th ∈ (0,1)` = fraction ảnh đầu vào.
- **Loss**:
  - Objectness: Focal Loss (α=0.25, γ=2.0) — chống dominance của negative cells.
  - Class: BCE multi-label ở positive cells.
  - Box: 1 − GIoU.
- **Scale assignment**: object area chuẩn hoá < 0.05² → P3, < 0.05 → P4, lớn hơn → P5.

## Notes

- Pretrained ResNet-34 → backbone output channels (128, 256, 512) khớp với neck mặc định, không cần config thêm.
- Vô hiệu hoá pretrained khi load checkpoint (đã set trong eval scripts).
- Round-trip encode/decode đã verify pixel-perfect (xem [losses.py](../utils/losses.py) + [test_and_visualize.py](../../evaluation/test_and_visualize.py)).
