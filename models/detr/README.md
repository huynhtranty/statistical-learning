# DETR (DEtection TRansformer)

## Mô tả

DETR là mô hình **transformer-based detector** tiên phong, loại bỏ hoàn toàn
các thành phần thủ công như anchor boxes và Non-Maximum Suppression (NMS).

Kiến trúc gồm 4 phần chính:

1. **CNN Backbone** (ResNet-50): trích xuất feature map từ ảnh.
2. **Transformer Encoder**: xử lý feature map kèm positional encoding.
3. **Transformer Decoder**: dùng object queries để dự đoán set of objects.
4. **Prediction Heads**: class head + bbox head cho mỗi query.

Sử dụng Hungarian matching để ghép cặp 1-1 giữa predictions và ground truth.

## Cấu trúc file

| File | Mô tả |
|------|-------|
| `model.py` | Hàm `build_detr()` — ghép backbone + transformer + heads thành model hoàn chỉnh |
| `backbone.py` | `DETRBackbone` — ResNet-50 + projection conv để trích xuất feature |
| `transformer.py` | `DETRTransformer` — encoder-decoder + positional encoding 2D |
| `matcher.py` | `HungarianMatcher` — ghép cặp predictions-targets bằng thuật toán Hungarian |
| `train.py` | Script kiểm tra kiến trúc và skeleton cho training |
| `config.yaml` | Cấu hình model cơ bản |
| `__init__.py` | Export các class/hàm chính |

## Cách chạy kiểm tra kiến trúc

```bash
# Từ thư mục gốc project (statistical-learning/)
python models/detr/train.py --num_classes 5
python models/detr/train.py --num_classes 5 --num_queries 100 --device cuda
```

Script sẽ:
- Build model DETR (backbone + transformer + heads)
- In ra số lượng tham số
- Forward thử với ảnh dummy 640×640
- In ra kích thước output (pred_logits, pred_boxes)

## Các phần chưa làm

- [ ] DataLoader với COCO annotations
- [ ] Training loop (AdamW, scheduler, gradient clipping)
- [ ] Tích hợp SetCriterion (loss) với HungarianMatcher
- [ ] Auxiliary loss (dự đoán ở mỗi decoder layer)
- [ ] Validation sau mỗi epoch
- [ ] Lưu checkpoint tốt nhất
- [ ] Hỗ trợ backbone khác (ResNet-101, ResNeXt)
- [ ] Load pretrained DETR từ HuggingFace
