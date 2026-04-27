# Faster R-CNN

## Mô tả

Faster R-CNN là mô hình **two-stage detector** kinh điển cho object detection:

1. **Region Proposal Network (RPN)** đề xuất các vùng có thể chứa vật thể.
2. **ROI Head** phân loại lớp và tinh chỉnh bounding box cho từng vùng đề xuất.

Sử dụng ResNet-50 + FPN làm backbone, khởi tạo từ trọng số pretrained trên COCO.

## Cấu trúc file

| File | Mô tả |
|------|-------|
| `model.py` | Hàm `build_faster_rcnn()` — tạo mô hình với số lớp tuỳ chỉnh |
| `backbone.py` | Hàm `build_backbone()` — tạo ResNet-50 + FPN backbone |
| `train.py` | Script kiểm tra kiến trúc và skeleton cho training |
| `config.yaml` | Cấu hình model cơ bản |
| `__init__.py` | Export các hàm chính |

## Cách chạy kiểm tra kiến trúc

```bash
# Từ thư mục gốc project (statistical-learning/)
python models/faster_rcnn/train.py --num_classes 6
python models/faster_rcnn/train.py --num_classes 6 --device cuda
```

Script sẽ:
- Build model Faster R-CNN
- In ra số lượng tham số
- Forward thử với ảnh dummy 640×640
- Báo kết quả thành công

## Các phần chưa làm

- [ ] DataLoader với COCO annotations
- [ ] Training loop (optimizer, scheduler, epoch loop)
- [ ] Validation sau mỗi epoch
- [ ] Lưu checkpoint tốt nhất
- [ ] Tuỳ chỉnh anchor sizes/aspect ratios
- [ ] Data augmentation
- [ ] Hỗ trợ backbone khác (ResNet-101, MobileNet, v.v.)
