#!/usr/bin/env python3
"""
Evaluation script that runs inference and saves images with drawn bounding boxes + class names.

Usage:
    # Auto-read config from model config.yaml
    python evaluation/test_and_visualize.py \
        --model faster_rcnn \
        --weights weights/faster_rcnn.pt \
        --data data/images/test \
        --output evaluation/results/test_vis \
        --device cuda

    # Override config values
    python evaluation/test_and_visualize.py \
        --model yolo \
        --weights models/yolo/checkpoints/best_model.pt \
        --data data/images/test \
        --output evaluation/results/test_vis \
        --device cuda \
        --conf-threshold 0.3
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.utils.coco_dataset import get_class_names, DEFAULT_CLASSES
from models.utils.losses import DEFAULT_YOLO_ANCHORS


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model_config(model_type: str) -> dict:
    """Load config.yaml for the specified model type."""
    config_path = PROJECT_ROOT / "models" / model_type / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Color palette: one color per class (BGR for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
_CLASS_COLORS = [
    (235, 122, 67),    # cat     – warm orange
    (67, 142, 219),    # dog     – sky blue
    (130, 235, 149),    # horse   – mint green
    (235, 176, 33),    # cow     – golden yellow
    (198, 120, 255),   # bird    – violet
    (129, 236, 236),   # sheep   – cyan
    (255, 200, 150),   # extra 1
    (180, 130, 90),    # extra 2
]


def get_color(class_id: int) -> tuple[int, int, int]:
    return _CLASS_COLORS[class_id % len(_CLASS_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# Bounding box drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_boxes(
    image: np.ndarray,
    boxes: list,
    labels: list,
    scores: list | None = None,
    class_names: list[str] | None = None,
    box_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw bounding boxes with class labels and confidence scores on an image.

    Args:
        image: RGB image as numpy array (H, W, 3)
        boxes: list of [x, y, w, h] in pixel coordinates (COCO format)
        labels: list of class indices (int)
        scores: list of confidence scores (float), optional
        class_names: list of class names; falls back to DEFAULT_CLASSES
        box_thickness: line thickness
        font_scale: font scale for label text

    Returns:
        Image with drawn boxes (RGB numpy array)
    """
    class_names = class_names or DEFAULT_CLASSES
    img_h, img_w = image.shape[:2]

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x, y, w, h = box
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Clamp to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

        color = get_color(int(label) % len(_CLASS_COLORS))

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

        # Build label text
        cls_name = class_names[int(label)] if int(label) < len(class_names) else f"class_{label}"
        if scores is not None and i < len(scores):
            label_text = f"{cls_name} {scores[i]:.2f}"
        else:
            label_text = cls_name

        # Text background (for readability)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, font_scale, 1
        )
        text_y = max(y1 - 4, text_h + 4)

        # Filled background rectangle
        cv2.rectangle(
            image,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline // 2),
            color,
            -1,
        )

        # White text
        cv2.putText(
            image,
            label_text,
            (x1, text_y - baseline // 2),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_type: str, weights_path: str, device: str, num_classes: int = 10) -> nn.Module:
    """Load custom-built models (ImageNet pretrained backbone, head trained on our dataset)."""
    if model_type == "faster_rcnn":
        from models.faster_rcnn.model import build_faster_rcnn
        # num_classes + 1 (cộng background). pretrained=False vì load từ checkpoint của ta.
        model = build_faster_rcnn(num_classes=num_classes + 1, pretrained=False)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    elif model_type == "yolo":
        from models.yolo.model import build_yolo
        # backbone weights đã có trong checkpoint → tránh download lại pretrained.
        model = build_yolo(num_classes=num_classes, pretrained_backbone=False)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    elif model_type == "detr":
        from models.detr.model import build_detr
        # HF wrapper: cấu trúc tự load từ HF hub khi build, sau đó load checkpoint của ta.
        model = build_detr(num_classes=num_classes, pretrained_coco=True)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image_pil(image_path: Path, input_size: int = 640) -> tuple:
    """Load image and return (PIL_image, original_size, tensor, scale_x, scale_y, pad_x, pad_y).

    Phải đồng bộ với pipeline training trong models/utils/coco_dataset.py:
    direct resize tới (input_size, input_size) — KHÔNG letterbox.
    """
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    resized = pil_img.resize((input_size, input_size), Image.BILINEAR)

    arr = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)

    # Trả về scale_x / scale_y để map predictions từ input-space về ảnh gốc.
    scale_x = input_size / float(orig_w)
    scale_y = input_size / float(orig_h)
    pad_x = 0
    pad_y = 0
    return pil_img, (orig_w, orig_h), tensor, (scale_x, scale_y), pad_x, pad_y


def _nms(boxes: list, scores: list, labels: list,
          iou_threshold: float = 0.45,
          class_agnostic_iou: float = 0.7,
          top_k: int | None = None,
          min_area_frac: float = 0.0,
          img_wh: tuple[int, int] | None = None) -> tuple[list, list, list]:
    """Per-class NMS + class-agnostic NMS + top-K + min-area cleanup.

    Untrained / under-trained YOLO heads spray many spatially-distinct
    low-confidence boxes — per-class NMS alone leaves them all because they
    don't overlap. Layering a class-agnostic NMS (higher IoU) collapses
    cross-class duplicates, then top-K caps output so the rendered image
    stays readable.
    """
    if not boxes:
        return [], [], []

    import torch
    import torchvision.ops

    # Drop tiny boxes (visual noise from grid-cell predictions)
    if min_area_frac > 0 and img_wh is not None:
        img_w, img_h = img_wh
        min_area = float(img_w) * float(img_h) * min_area_frac
        kept = [(b, s, l) for b, s, l in zip(boxes, scores, labels)
                if b[2] * b[3] >= min_area]
        if not kept:
            return [], [], []
        boxes = [k[0] for k in kept]
        scores = [k[1] for k in kept]
        labels = [k[2] for k in kept]

    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    xyxy = torch.stack(
        (boxes_t[:, 0], boxes_t[:, 1],
         boxes_t[:, 0] + boxes_t[:, 2],
         boxes_t[:, 1] + boxes_t[:, 3]),
        dim=1,
    )

    keep = torchvision.ops.batched_nms(xyxy, scores_t, labels_t, iou_threshold)

    if class_agnostic_iou is not None and class_agnostic_iou > 0 and keep.numel() > 1:
        keep2 = torchvision.ops.nms(xyxy[keep], scores_t[keep], class_agnostic_iou)
        keep = keep[keep2]

    # batched_nms returns sorted-by-score; top-K cap keeps the strongest.
    if top_k is not None and keep.numel() > top_k:
        keep = keep[:top_k]

    idx_list = keep.tolist()
    return ([boxes[i] for i in idx_list],
            [scores[i] for i in idx_list],
            [labels[i] for i in idx_list])


def _box_iou(a: list, b: list) -> float:
    """IoU between [x,y,w,h] boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a2x, a2y = ax + aw, ay + ah
    b2x, b2y = bx + bw, by + bh
    xi1, yi1 = max(ax, bx), max(ay, by)
    xi2, yi2 = min(a2x, b2x), min(a2y, b2y)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _decode_yolo_single_scale(
    pred: torch.Tensor,
    orig_w: int,
    orig_h: int,
    scale_x: float,
    scale_y: float,
    pad_x: int,
    pad_y: int,
    input_size: int,
    conf_threshold: float,
    num_classes: int = 10,
    num_anchors: int = 3,
) -> tuple[list, list, list]:
    """Vectorised decode for one YOLO scale → list of [x,y,w,h] boxes in original-image pixels."""
    # pred: (1, A*(5+C), H, W) → batch 0
    pred = pred[0]  # (C_total, H, W)
    H, W = pred.shape[1], pred.shape[2]
    stride_x = input_size / W
    stride_y = input_size / H

    # Reshape to (A, 5+C, H, W)
    pred = pred.view(num_anchors, 5 + num_classes, H, W)
    scale_idx = int(round(np.log2(input_size / max(int(H), 1)))) - 3
    scale_idx = max(0, min(scale_idx, len(DEFAULT_YOLO_ANCHORS) - 1))
    anchors = torch.tensor(DEFAULT_YOLO_ANCHORS[scale_idx], dtype=pred.dtype, device=pred.device)

    # YOLOv5 parameterization:
    #   pred_xy = 2*sigmoid(t) - 0.5  ∈ [-0.5, 1.5]   (offset trong cell, hỗ trợ neighbor)
    #   pred_wh = (2*sigmoid(t))^2    ∈ [0, 4]        (fraction of image normalized)
    tx = 2.0 * pred[:, 0, :, :].sigmoid() - 0.5
    ty = 2.0 * pred[:, 1, :, :].sigmoid() - 0.5
    tw = (2.0 * pred[:, 2, :, :].sigmoid()).pow(2) * anchors[:, 0].view(num_anchors, 1, 1)
    th = (2.0 * pred[:, 3, :, :].sigmoid()).pow(2) * anchors[:, 1].view(num_anchors, 1, 1)
    obj = pred[:, 4, :, :].sigmoid()
    cls = pred[:, 5:, :, :].sigmoid()  # (A, C, H, W)

    # score per (A, C, H, W) = obj * cls
    scores_full = obj.unsqueeze(1) * cls  # (A, C, H, W)
    # Best class per (A, H, W)
    best_score, best_cls = scores_full.max(dim=1)  # (A, H, W)

    keep = best_score > conf_threshold
    if not keep.any():
        return [], [], []

    a_idx, j_idx, i_idx = torch.where(keep)  # anchor, row(cy), col(cx)
    sel_score = best_score[a_idx, j_idx, i_idx].cpu().tolist()
    sel_cls = best_cls[a_idx, j_idx, i_idx].cpu().tolist()
    sel_tx = tx[a_idx, j_idx, i_idx].cpu().numpy()
    sel_ty = ty[a_idx, j_idx, i_idx].cpu().numpy()
    sel_tw = tw[a_idx, j_idx, i_idx].cpu().numpy()
    sel_th = th[a_idx, j_idx, i_idx].cpu().numpy()
    cx_grid = i_idx.cpu().numpy()
    cy_grid = j_idx.cpu().numpy()

    boxes, scores, labels = [], [], []
    for k in range(len(sel_score)):
        # xy: offset trong cell × stride → pixel position trong input 640
        xc_in = (cx_grid[k] + float(sel_tx[k])) * stride_x
        yc_in = (cy_grid[k] + float(sel_ty[k])) * stride_y
        # wh: fraction of image × input_size
        bw_in = float(sel_tw[k]) * input_size
        bh_in = float(sel_th[k]) * input_size

        x1_in = xc_in - bw_in / 2
        y1_in = yc_in - bh_in / 2

        # Inverse of direct resize: pixel_in_orig = (pixel_in_input - pad) / scale_axis
        x = (x1_in - pad_x) / scale_x
        y = (y1_in - pad_y) / scale_y
        w = bw_in / scale_x
        h = bh_in / scale_y

        x = max(0.0, min(x, orig_w))
        y = max(0.0, min(y, orig_h))
        w = min(w, orig_w - x)
        h = min(h, orig_h - y)
        if w <= 0 or h <= 0:
            continue

        boxes.append([x, y, w, h])
        scores.append(float(sel_score[k]))
        labels.append(int(sel_cls[k]))

    return boxes, scores, labels


def run_predictions(
    model: nn.Module,
    model_type: str,
    image_path: Path,
    device: str,
    input_size: int = 640,
    conf_threshold: float = 0.5,
    top_k: int | None = 30,
    min_area_frac: float = 0.001,
) -> tuple[list, list, list]:
    """Run model on one image and return (boxes, labels, scores) in pixel coords."""
    pil_img, orig_size, tensor, scales, pad_x, pad_y = preprocess_image_pil(
        image_path, input_size
    )
    orig_w, orig_h = orig_size
    scale_x, scale_y = scales

    with torch.no_grad():
        if model_type == "faster_rcnn":
            output = model([tensor.to(device)])[0]
            boxes_xyxy = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            result_boxes, result_labels, result_scores = [], [], []
            for bx, sc, lb in zip(boxes_xyxy, scores, labels):
                if sc < conf_threshold:
                    continue
                # Faster R-CNN trả label 1..N (0 = background) → đổi về 0..N-1 cho class_names.
                lb_idx = int(lb) - 1
                if lb_idx < 0:
                    continue
                x1, y1, x2, y2 = bx
                x1 = (x1 - pad_x) / scale_x
                y1 = (y1 - pad_y) / scale_y
                x2 = (x2 - pad_x) / scale_x
                y2 = (y2 - pad_y) / scale_y
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                result_boxes.append([x1, y1, w, h])
                result_labels.append(lb_idx)
                result_scores.append(float(sc))

        elif model_type == "yolo":
            # Import decode function from train.py to use same decode logic
            from models.yolo.train import decode_predictions as yolo_decode
            from models.yolo.train import nms as yolo_nms

            # YOLO returns list of 3 tensors at different scales (P3,P4,P5)
            predictions_list = model(tensor.unsqueeze(0).to(device))
            if not isinstance(predictions_list, list):
                predictions_list = [predictions_list]

            # Số class lấy từ shape output
            ch = predictions_list[0].shape[1]
            num_classes_decode = ch // 3 - 5

            # Use same decode as train.py
            decoded = yolo_decode(
                predictions_list,
                conf_threshold=conf_threshold,
                num_classes=num_classes_decode,
                img_size=input_size,
            )

            # Convert to draw format [x, y, w, h] and apply NMS
            all_boxes, all_scores, all_labels = [], [], []
            for preds in decoded:
                for p in preds:
                    bbox = p['bbox']  # [x1, y1, x2, y2]
                    x = (bbox[0] - pad_x) / scale_x
                    y = (bbox[1] - pad_y) / scale_y
                    w = (bbox[2] - bbox[0]) / scale_x
                    h = (bbox[3] - bbox[1]) / scale_y
                    x = max(0.0, min(x, orig_w))
                    y = max(0.0, min(y, orig_h))
                    w = min(w, orig_w - x)
                    h = min(h, orig_h - y)
                    if w > 0 and h > 0:
                        all_boxes.append([x, y, w, h])
                        all_scores.append(p['score'])
                        all_labels.append(p['class'])

            # Per-class NMS + class-agnostic NMS + top-K + min-area cleanup
            # to keep visualisation readable even with an under-trained model.
            result_boxes, result_scores, result_labels = _nms(
                all_boxes, all_scores, all_labels,
                iou_threshold=0.45,
                class_agnostic_iou=0.7,
                top_k=top_k,
                min_area_frac=min_area_frac,
                img_wh=(orig_w, orig_h),
            )

        elif model_type == "detr":
            # Custom DETR takes a 4D tensor (B, 3, H, W).
            output = model(tensor.unsqueeze(0).to(device))
            probas = output["pred_logits"].softmax(-1)
            boxes_norm = output["pred_boxes"].cpu().numpy()

            result_boxes, result_labels, result_scores = [], [], []
            for logits, box in zip(probas.squeeze(0), boxes_norm.squeeze(0)):
                scores, labels = logits[:-1].max(0)
                if scores.item() < conf_threshold:
                    continue
                cx, cy, bw, bh = box
                x = (cx - bw / 2) * orig_w
                y = (cy - bh / 2) * orig_h
                w = bw * orig_w
                h = bh * orig_h
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = min(w, orig_w - x)
                h = min(h, orig_h - y)
                if w <= 0 or h <= 0:
                    continue
                result_boxes.append([x, y, w, h])
                result_labels.append(int(labels.item()))
                result_scores.append(float(scores.item()))

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return result_boxes, result_labels, result_scores


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference + save images with drawn bounding boxes"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["faster_rcnn", "yolo", "detr"],
                        help="Model type")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--data", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for visualized images")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"],
                        help="Device to run inference on")
    parser.add_argument("--conf-threshold", type=float, default=None,
                        help="Confidence threshold (default: from config.yaml)")
    parser.add_argument("--input-size", type=int, default=None,
                        help="Input image size (default: from config.yaml)")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Max boxes per image after NMS (visual clutter cap)")
    parser.add_argument("--min-area-frac", type=float, default=0.001,
                        help="Drop boxes smaller than this fraction of image area")
    parser.add_argument("--show-gt", action="store_true",
                        help="Also draw ground truth boxes (requires --ann-file)")
    parser.add_argument("--ann-file", type=str, default=None,
                        help="Path to COCO annotation JSON (for ground truth)")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save predictions JSON")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of classes (default: from config.yaml)")
    return parser.parse_args()


def load_gt_annotations(ann_file: Path) -> dict:
    with open(ann_file) as f:
        return json.load(f)


def build_gt_index(ann_data: dict) -> dict:
    """Pre-build index: image_id -> list of annotations for fast lookup."""
    index = {}
    for ann in ann_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in index:
            index[img_id] = []
        index[img_id].append(ann)
    return index


def build_img_name_to_id(ann_data: dict) -> dict:
    """Pre-build index: file_name -> image_id."""
    return {img["file_name"]: img["id"] for img in ann_data.get("images", [])}


def _build_cat_id_to_cls_idx(
    ann_data: dict, class_names: list[str]
) -> dict[int, int]:
    """Map category_id (COCO, có thể không liền mạch) → 0-based class index theo class_names.

    Ưu tiên ghép qua TÊN category — robust với mọi COCO scheme (1-based, 0-based,
    hoặc giữ ID 17/18/.. như COCO 2017).
    """
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    mapping: dict[int, int] = {}
    for cat in ann_data.get("categories", []):
        cat_id = cat.get("id")
        name = cat.get("name")
        if cat_id is None or name is None:
            continue
        if name in name_to_idx:
            mapping[int(cat_id)] = name_to_idx[name]
    return mapping


def get_gt_for_image(
    ann_data: dict,
    image_file_name: str,
    class_names: list[str],
    img_name_to_id: dict | None = None,
    gt_index: dict | None = None,
    cat_mapping: dict | None = None,
) -> tuple[list, list]:
    """Extract GT boxes/labels for a given image file name.

    Uses pre-built indexes for O(1) lookup instead of O(n) loops.
    """
    # Fast lookup using pre-built index
    if img_name_to_id is not None and gt_index is not None and cat_mapping is not None:
        img_id = img_name_to_id.get(image_file_name)
        if img_id is None:
            return [], []
        anns = gt_index.get(img_id, [])
    else:
        # Fallback: slow lookup
        img_id = None
        for img in ann_data.get("images", []):
            if img["file_name"] == image_file_name:
                img_id = img["id"]
                break
        if img_id is None:
            return [], []
        anns = ann_data.get("annotations", [])
        anns = [a for a in anns if a["image_id"] == img_id]
        if cat_mapping is None:
            cat_mapping = _build_cat_id_to_cls_idx(ann_data, class_names)

    boxes, labels = [], []
    for ann in anns:
        cat_id = int(ann["category_id"])
        cls_idx = (cat_mapping or {}).get(cat_id)
        if cls_idx is None or cls_idx < 0 or cls_idx >= len(class_names):
            continue
        x, y, w, h = ann["bbox"]
        boxes.append([float(x), float(y), float(w), float(h)])
        labels.append(cls_idx)

    return boxes, labels


def main():
    args = parse_args()

    # Load config from yaml
    config = load_model_config(args.model)
    model_config = config.get("model", {})
    inference_config = config.get("inference", {})
    training_config = config.get("training", {})

    # Use config values as defaults, CLI args override
    num_classes = args.num_classes if args.num_classes is not None else model_config.get("num_classes", 10)
    input_size = args.input_size if args.input_size is not None else model_config.get("image_size", 640)
    conf_threshold = args.conf_threshold if args.conf_threshold is not None else inference_config.get("conf_threshold", 0.5)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    # Load class names from config or fallback
    if "classes" in model_config:
        class_names = model_config["classes"]
        # Remove __background__ if present (for faster_rcnn indexing)
        if class_names and class_names[0] == "__background__":
            class_names = class_names[1:]
    else:
        class_names = get_class_names()
    print(f"[Info] Classes ({len(class_names)}): {class_names}")
    print(f"[Info] Config: input_size={input_size}, conf_threshold={conf_threshold}, num_classes={num_classes}")

    # Load model
    print(f"[Info] Loading {args.model} from {args.weights}...")
    model = load_model(args.model, args.weights, device, num_classes)

    # Load GT annotations if requested
    gt_data = None
    gt_index = None
    img_name_to_id = None
    cat_mapping = None
    if args.show_gt and args.ann_file:
        print(f"[Info] Loading GT annotations from {args.ann_file}")
        gt_data = load_gt_annotations(Path(args.ann_file))
        # Pre-build indexes for O(1) lookup
        gt_index = build_gt_index(gt_data)
        img_name_to_id = build_img_name_to_id(gt_data)
        cat_mapping = _build_cat_id_to_cls_idx(gt_data, class_names)
        print(f"[Info] Built GT index with {len(gt_index)} images")

    # Find test images
    image_dir = Path(args.data)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix in extensions])

    if not image_files:
        print(f"[Error] No images found in {image_dir}")
        return

    print(f"[Info] Found {len(image_files)} images")

    if args.max_images:
        image_files = image_files[: args.max_images]

    # Output directory
    out_dir = Path(args.output)
    if out_dir.is_file():
        out_dir = out_dir.parent / (out_dir.stem + "_vis")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Output directory: {out_dir}")

    all_predictions = []
    processed = 0

    for img_path in image_files:
        # Run prediction
        boxes, labels, scores = run_predictions(
            model, args.model, img_path, device,
            input_size=input_size, conf_threshold=conf_threshold,
            top_k=args.top_k, min_area_frac=args.min_area_frac,
        )

        # Sort by score descending — strongest detections drawn last, so their
        # labels stay readable on top of any weaker overlapping boxes.
        if scores:
            order = sorted(range(len(scores)), key=lambda i: scores[i])
            boxes = [boxes[i] for i in order]
            labels = [labels[i] for i in order]
            scores = [scores[i] for i in order]

        # Load original image (no resize, no padding) for drawing
        orig_img = Image.open(img_path).convert("RGB")
        img_rgb = np.array(orig_img)

        # Draw predictions
        img_rgb = draw_boxes(
            img_rgb, boxes, labels, scores,
            class_names=class_names,
        )

        # Draw GT if available
        if gt_data is not None:
            gt_boxes, gt_labels = get_gt_for_image(
                gt_data, img_path.name, class_names,
                img_name_to_id, gt_index, cat_mapping
            )
            # Draw GT with dashed lines (simulate with thinner boxes, distinct color)
            # We reuse draw_boxes but skip scores and use a "GT" label
            gt_draw_boxes, gt_draw_labels = [], []
            for gb, gl in zip(gt_boxes, gt_labels):
                # Label = class idx + offset to distinguish visually
                gt_draw_boxes.append(gb)
                gt_draw_labels.append(gl)
            # Draw GT with thin dashed green boxes (positive offset)
            img_rgb = draw_boxes_gt(
                img_rgb, gt_boxes, gt_labels,
                class_names=class_names,
            )

        # Save image
        out_path = out_dir / f"{img_path.stem}_result.jpg"
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Store predictions for JSON
        all_predictions.append({
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "predictions": [
                {"bbox": b, "score": s, "class": l}
                for b, s, l in zip(boxes, scores, labels)
            ],
        })

        processed += 1
        status = f"{img_path.name}: {len(boxes)} detections"
        if gt_data is not None:
            status += f", {len(gt_boxes)} GT boxes"
        print(f"  [{processed}/{len(image_files)}] {status}")

    # Save predictions JSON
    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_predictions, f, indent=2)
        print(f"[Info] Predictions saved to {save_path}")

    print(f"\n[Done] {processed} images processed → {out_dir}")


def draw_boxes_gt(
    image: np.ndarray,
    boxes: list,
    labels: list,
    class_names: list[str] | None = None,
    box_thickness: int = 1,
) -> np.ndarray:
    """
    Draw ground truth boxes with dashed/dotted style and green tint.
    Label format: class name only (no score for GT).
    """
    class_names = class_names or DEFAULT_CLASSES
    img_h, img_w = image.shape[:2]

    for box, label in zip(boxes, labels):
        x, y, w, h = box
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

        color = get_color(int(label) % len(_CLASS_COLORS))

        # Draw dashed rectangle using line segments
        def draw_dashed_rect(img, pt1, pt2, color, thickness):
            x1, y1 = pt1
            x2, y2 = pt2
            dash = 8
            gap = 4
            # Top
            for x in range(x1, x2, dash + gap):
                cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
            # Bottom
            for x in range(x1, x2, dash + gap):
                cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
            # Left
            for y in range(y1, y2, dash + gap):
                cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
            # Right
            for y in range(y1, y2, dash + gap):
                cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)

        draw_dashed_rect(image, (x1, y1), (x2, y2), color, box_thickness)

        # GT label
        cls_name = class_names[int(label)] if int(label) < len(class_names) else f"class_{label}"
        label_text = f"GT: {cls_name}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)
        text_y = max(y1 - 4, text_h + 4)

        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline // 2),
            color,
            -1,
        )
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv2.putText(
            image, label_text,
            (x1, text_y - baseline // 2),
            font, font_scale,
            (255, 255, 255),
            1, cv2.LINE_AA,
        )

    return image


if __name__ == "__main__":
    main()
