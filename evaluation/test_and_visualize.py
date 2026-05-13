#!/usr/bin/env python3
"""
Evaluation script that runs inference and saves images with drawn bounding boxes + class names.

Usage:
    python evaluation/test_and_visualize.py \
        --model faster_rcnn \
        --weights weights/faster_rcnn.pt \
        --data data/images/test \
        --output evaluation/results/test_vis \
        --device cuda \
        --conf-threshold 0.5 \
        --show-gt

    python evaluation/test_and_visualize.py \
        --model yolo \
        --weights models/yolo/checkpoints/best_model.pt \
        --data data/images/test \
        --output evaluation/results/test_vis \
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.utils.coco_dataset import get_class_names, DEFAULT_CLASSES

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
    if model_type == "faster_rcnn":
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

    elif model_type == "yolo":
        from models.yolo.model import build_yolo
        model = build_yolo(num_classes=num_classes)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    elif model_type == "detr":
        from torchvision.models.detection import detr_resnet50, DetrResNet50_Weights
        model = detr_resnet50(weights=DetrResNet50_Weights.DEFAULT)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image_pil(image_path: Path, input_size: int = 640) -> tuple:
    """Load image and return (PIL_image, original_size, tensor, scale)."""
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Resize to input_size keeping aspect ratio (letterbox)
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # Pad to square
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    # Convert to tensor [0,1]
    arr = np.array(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return pil_img, (orig_w, orig_h), tensor, scale, pad_x, pad_y


def _nms(boxes: list, scores: list, labels: list,
          iou_threshold: float = 0.45) -> tuple[list, list, list]:
    """Simple per-class NMS."""
    if not boxes:
        return [], [], []

    # Group by class
    from collections import defaultdict
    by_class = defaultdict(list)
    for b, s, l in zip(boxes, scores, labels):
        by_class[l].append((b, s))

    nms_boxes, nms_scores, nms_labels = [], [], []
    for cls_id, items in by_class.items():
        # Sort by score
        items = sorted(items, key=lambda x: x[1], reverse=True)
        keep = []
        while items:
            best = items.pop(0)
            keep.append(best)
            items = [
                (b, s) for b, s in items
                if _box_iou(best[0], b) < iou_threshold
            ]
        for b, s in keep:
            nms_boxes.append(b)
            nms_scores.append(s)
            nms_labels.append(cls_id)

    return nms_boxes, nms_scores, nms_labels


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
    scale: float,
    pad_x: int,
    pad_y: int,
    input_size: int,
    conf_threshold: float,
) -> tuple[list, list, list]:
    """Decode predictions from a single YOLO scale into pixel bboxes."""
    # pred: (batch, num_anchors*(5+num_classes), H, W) → take batch 0
    pred = pred[0]  # (C, H, W)
    C = pred.shape[0]
    H, W = pred.shape[1], pred.shape[2]
    num_classes = C // 3 - 5  # 3 anchors per cell
    stride = input_size / W   # grid cell size in input-space

    boxes, scores, labels = [], [], []
    for cy in range(H):
        for cx in range(W):
            for a in range(3):
                offset = a * (5 + num_classes)
                tx, ty, tw, th = pred[offset:offset+4, cy, cx].cpu().numpy()
                obj = float(pred[offset + 4, cy, cx].cpu())
                cls_logits = pred[offset+5:offset+5+num_classes, cy, cx].cpu().numpy()

                if obj < conf_threshold:
                    continue

                # sigmoid for class prob
                cls_probs = 1 / (1 + np.exp(-cls_logits))
                max_cls_prob = float(cls_probs.max())
                final_score = obj * max_cls_prob
                if final_score < conf_threshold:
                    continue

                pred_cls = int(cls_probs.argmax())

                # Convert tx,ty,tw,th to pixel coords
                xc = (cx + (1 / (1 + np.exp(-tx)))) * stride
                yc = (cy + (1 / (1 + np.exp(-ty)))) * stride
                bw = float(np.exp(tw)) * stride * 4
                bh = float(np.exp(th)) * stride * 4

                # Remove padding & scale back to original
                x = (xc - pad_x) / scale
                y = (yc - pad_y) / scale
                w = bw / scale
                h = bh / scale

                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = min(w, orig_w - x)
                h = min(h, orig_h - y)
                if w <= 0 or h <= 0:
                    continue

                boxes.append([x, y, w, h])
                scores.append(final_score)
                labels.append(pred_cls)

    return boxes, scores, labels


def run_predictions(
    model: nn.Module,
    model_type: str,
    image_path: Path,
    device: str,
    input_size: int = 640,
    conf_threshold: float = 0.5,
) -> tuple[list, list, list]:
    """Run model on one image and return (boxes, labels, scores) in pixel coords."""
    pil_img, orig_size, tensor, scale, pad_x, pad_y = preprocess_image_pil(
        image_path, input_size
    )
    orig_w, orig_h = orig_size

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
                x1, y1, x2, y2 = bx
                x1 = (x1 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                x2 = (x2 - pad_x) / scale
                y2 = (y2 - pad_y) / scale
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                result_boxes.append([x1, y1, w, h])
                result_labels.append(int(lb))
                result_scores.append(float(sc))

        elif model_type == "yolo":
            # YOLO returns list of 3 tensors at different scales (P3,P4,P5)
            predictions_list = model(tensor.unsqueeze(0).to(device))
            if not isinstance(predictions_list, list):
                predictions_list = [predictions_list]

            all_boxes, all_scores, all_labels = [], [], []
            for pred in predictions_list:
                b, s, l = _decode_yolo_single_scale(
                    pred, orig_w, orig_h,
                    scale, pad_x, pad_y,
                    input_size, conf_threshold,
                )
                all_boxes.extend(b)
                all_scores.extend(s)
                all_labels.extend(l)

            # NMS across all scales
            result_boxes, result_scores, result_labels = _nms(
                all_boxes, all_scores, all_labels
            )

        elif model_type == "detr":
            output = model([tensor.to(device)])
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
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Confidence threshold for predictions")
    parser.add_argument("--input-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--show-gt", action="store_true",
                        help="Also draw ground truth boxes (requires --ann-file)")
    parser.add_argument("--ann-file", type=str, default=None,
                        help="Path to COCO annotation JSON (for ground truth)")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save predictions JSON")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="Number of classes in the model")
    return parser.parse_args()


def load_gt_annotations(ann_file: Path) -> dict:
    with open(ann_file) as f:
        return json.load(f)


def get_gt_for_image(
    ann_data: dict,
    image_file_name: str,
    class_names: list[str],
) -> tuple[list, list]:
    """Extract GT boxes/labels for a given image file name."""
    # Find image id
    img_id = None
    for img in ann_data["images"]:
        if img["file_name"] == image_file_name:
            img_id = img["id"]
            break
    if img_id is None:
        return [], []

    # Map category_id -> class index (category_id in COCO starts at 1)
    boxes, labels = [], []
    for ann in ann_data["annotations"]:
        if ann["image_id"] != img_id:
            continue
        cat_id = ann["category_id"]  # 1-based
        cls_idx = cat_id - 1          # 0-based index
        if cls_idx < 0 or cls_idx >= len(class_names):
            continue
        x, y, w, h = ann["bbox"]
        boxes.append([float(x), float(y), float(w), float(h)])
        labels.append(cls_idx)

    return boxes, labels


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    # Load class names
    class_names = get_class_names()
    print(f"[Info] Classes: {class_names}")

    # Load model
    print(f"[Info] Loading {args.model} from {args.weights}...")
    model = load_model(args.model, args.weights, device, args.num_classes)

    # Load GT annotations if requested
    gt_data = None
    if args.show_gt and args.ann_file:
        print(f"[Info] Loading GT annotations from {args.ann_file}")
        gt_data = load_gt_annotations(Path(args.ann_file))

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
            input_size=args.input_size, conf_threshold=args.conf_threshold,
        )

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
                gt_data, img_path.name, class_names
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
