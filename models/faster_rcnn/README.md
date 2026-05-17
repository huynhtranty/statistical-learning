# Faster R-CNN

## Mô tả

Faster R-CNN là mô hình **two-stage detector** kinh điển cho object detection:

1. **Region Proposal Network (RPN)** đề xuất các vùng có thể chứa vật thể.
2. **ROI Head** phân loại lớp và tinh chỉnh bounding box cho từng vùng đề xuất.

Sử dụng `torchvision.models.detection.fasterrcnn_resnet50_fpn` với **trọng số pretrained trên COCO**, chỉ thay box predictor cuối cho `num_classes` của bài toán.

## Cấu trúc file

| File | Mô tả |
|------|-------|
| `model.py` | `build_faster_rcnn(num_classes, pretrained=True)` — tạo model và thay box predictor |
| `train.py` | Training loop hoàn chỉnh (RPN + ROI loss tự xử lý trong forward) |
| `config.yaml` | Cấu hình model cơ bản |

## Convention quan trọng

**`num_classes` BAO GỒM background** (label 0). Với 10 class foreground:
- Build model với `num_classes=11`.
- Dataset trả label 0..9 → train script tự cộng `+1` thành label 1..10.
- Evaluation/visualization tự trừ `-1` ngược lại.

(Logic này đã wire sẵn trong `train.py` và `evaluation/*.py`, không cần làm thủ công.)

## Cách chạy training

```bash
# Train (mặc định: COCO pretrained, replace box predictor cho 10 fg + 1 bg)
python models/faster_rcnn/train.py --data_root data --epochs 10 --batch_size 4 \
  --output weights/faster_rcnn.pt --device cuda

# Train với augmentation
python models/faster_rcnn/train.py --data_root data --epochs 20 --batch_size 4 \
  --augment --output weights/faster_rcnn.pt --device cuda

# Resume
python models/faster_rcnn/train.py --resume weights/faster_rcnn.pt --epochs 10 --device cuda
```

Checkpoints lưu ở `models/faster_rcnn/checkpoints/best_model.pt`, logs ở `models/faster_rcnn/logs/`.

## Cách inference / visualize

```bash
python evaluation/test_and_visualize.py --model faster_rcnn \
  --weights weights/faster_rcnn.pt \
  --data data/images/test --output evaluation/results/faster_rcnn_vis \
  --device cuda --conf-threshold 0.5 \
  --show-gt --ann-file data/annotations/test.json
```

## Notes

- COCO 91-class predictor được drop, predictor mới `FastRCNNPredictor(in_features, 11)` train từ đầu.
- Backbone ResNet-50 + FPN giữ nguyên COCO weights. Mặc định 3 stage cuối backbone được trainable (`trainable_backbone_layers=3`).
- Vì pretrained trên COCO (vốn có cat/dog/horse/cow/bird/sheep/elephant/bear/zebra/giraffe), model converge cực nhanh — 5-10 epochs đã có mAP đáng kể.
- Khi load checkpoint của ta (file `weights/faster_rcnn.pt`), eval scripts dùng `pretrained=False` để tránh download lại COCO weights.
