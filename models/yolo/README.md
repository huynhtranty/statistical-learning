# YOLO (You Only Look Once)

## Mô tả

YOLO là mô hình **one-stage detector** nổi tiếng với tốc độ inference nhanh.
Kiến trúc này là phiên bản đơn giản hoá, gồm 3 phần chính:

1. **Backbone** (DarkNet-style với CSP blocks): trích xuất feature ở 3 scale.
2. **Neck** (FPN-style): kết hợp feature map từ nhiều scale.
3. **Detection Head**: dự đoán bounding box, objectness score, và class logits.

## Cấu trúc file

| File | Mô tả |
|------|-------|
| `model.py` | Hàm `build_yolo()` — ghép backbone + neck + head thành model hoàn chỉnh |
| `backbone.py` | `YOLOBackbone` — trích xuất feature map ở 3 scale (stride 8, 16, 32) |
| `neck.py` | `YOLONeck` — FPN top-down pathway kết hợp feature |
| `head.py` | `YOLODetectionHead` — dự đoán detection ở mỗi scale |
| `train.py` | Script kiểm tra kiến trúc và skeleton cho training |
| `config.yaml` | Cấu hình model cơ bản |
| `__init__.py` | Export các class/hàm chính |

## Cách chạy kiểm tra kiến trúc

```bash
# Từ thư mục gốc project (statistical-learning/)
python models/yolo/train.py --num_classes 5
python models/yolo/train.py --num_classes 5 --device cuda
```

Script sẽ:
- Build model YOLO (backbone + neck + head)
- In ra số lượng tham số
- Forward thử với ảnh dummy 640×640
- In ra kích thước output ở mỗi scale

## Các phần chưa làm

- [ ] Định nghĩa anchor sizes cụ thể
- [ ] Decode predictions (chuyển raw output thành bounding box thật)
- [ ] Non-Maximum Suppression (NMS)
- [ ] Loss function hoàn chỉnh (objectness + cls + bbox)
- [ ] DataLoader với annotations
- [ ] Training loop
- [ ] Pretrained backbone
- [ ] Data augmentation (mosaic, mixup, v.v.)
- [ ] Bottom-up pathway (PAN) trong Neck
