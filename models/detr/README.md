# DETR (DEtection TRansformer)

## Mô tả

DETR là mô hình **transformer-based detector** tiên phong, loại bỏ hoàn toàn
các thành phần thủ công như anchor boxes và Non-Maximum Suppression (NMS).

Kiến trúc gồm 4 phần chính:

1. **CNN Backbone** (ResNet-50) trích xuất feature map.
2. **Transformer Encoder** xử lý feature map kèm 2D positional encoding.
3. **Transformer Decoder** dùng `num_queries` object queries để dự đoán set of objects.
4. **Prediction Heads**: class head (`num_classes + 1` cho "no object") + bbox head (cxcywh chuẩn hoá [0,1]).

Training dùng **Hungarian matching** (set prediction loss): ghép cặp 1-1 giữa predictions và targets.

## Hai biến thể trong project

| Biến thể | Khi nào dùng | Pretrained source |
|---|---|---|
| **HuggingFace DETR** (mặc định) | Train + Eval thực tế | `facebook/detr-resnet-50` (COCO pretrained, drop class head 91→11) |
| **DETR custom** | Báo cáo phần "tự build" | Chỉ ResNet-50 ImageNet backbone, transformer + heads random |

Chọn biến thể qua `build_detr(pretrained_coco=True/False)`. Cả 2 cùng expose interface `{"pred_logits", "pred_boxes"}` nên downstream (matcher + SetCriterion + eval) đồng nhất.

## Cấu trúc file

| File | Mô tả |
|------|-------|
| `model.py` | `build_detr()` — chọn HF wrapper hoặc custom DETR; `HuggingFaceDETR` tự normalize ImageNet bên trong |
| `backbone.py` | `DETRBackbone` — ResNet-50 + 1x1 proj cho biến thể custom |
| `transformer.py` | `DETRTransformer` — encoder/decoder + 2D positional encoding |
| `matcher.py` | `HungarianMatcher` — ghép cặp dựa trên cost (CE + L1 + GIoU) |
| `train.py` | Training loop với gradient clipping, AdamW |
| `config.yaml` | Cấu hình model (num_queries, hidden_dim, ...) |

## Cách chạy training

```bash
# Train (mặc định: HF facebook/detr-resnet-50 COCO pretrained)
python models/detr/train.py --data_root data --epochs 10 --batch_size 4 \
  --num_queries 100 --output weights/detr.pt --device cuda

# Train DETR custom from scratch (báo cáo phần educational)
# → Phải sửa build_detr trong train.py: pretrained_coco=False
```

Checkpoints lưu ở `models/detr/checkpoints/best_model.pt`, logs ở `models/detr/logs/`.

## Cách inference / visualize

```bash
python evaluation/test_and_visualize.py --model detr \
  --weights weights/detr.pt \
  --data data/images/test --output evaluation/results/detr_vis \
  --device cuda --conf-threshold 0.5 \
  --show-gt --ann-file data/annotations/test.json
```

## Convention dữ liệu

- **Targets cho training**: `{"labels": (N,), "boxes": (N, 4)}` với boxes ở định dạng **cxcywh chuẩn hoá [0,1]**.
  Hàm `prepare_targets(targets, device, img_size)` trong `train.py` tự convert từ dataset xyxy-pixel → cxcywh-normalized.
- **Class labels**: 0..N-1 (DETR đặt thêm class `N` ngầm cho "no object" qua `class_head` shape `num_classes+1`).
- **Image preprocessing**: HF wrapper tự normalize ImageNet mean/std bên trong; dataset chỉ cần xuất tensor [0,1] như các model khác.

## Dependencies

- `transformers>=4.38.0` (đã có trong `requirements.txt`).
- `timm` (cài tự động qua dependency của transformers; nếu thiếu: `pip install timm`).
