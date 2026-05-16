#!/usr/bin/env python3
"""
Generate predictions from a trained model for evaluation.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from models.utils.losses import DEFAULT_YOLO_ANCHORS


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions for evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=["faster_rcnn", "yolo", "detr"],
                        help="Model type")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--annotations", type=str, default=None,
                        help="COCO annotations JSON (for image_id/category_id mapping)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for predictions")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run inference on")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="Confidence threshold for predictions (default 0.25 — đồng bộ "
                             "với inference config; 0.5 quá khắt khe cho YOLO custom)")
    parser.add_argument("--nms-iou", type=float, default=0.45,
                        help="NMS IoU threshold (YOLO only)")
    parser.add_argument("--input-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="Number of classes in the model")
    return parser.parse_args()


def load_yolo_model(weights_path: str, device: str, num_classes: int = 10) -> nn.Module:
    from models.yolo.model import build_yolo
    model = build_yolo(num_classes=num_classes, pretrained_backbone=False)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model.to(device)


def load_faster_rcnn_model(weights_path: str, device: str, num_classes: int = 10) -> nn.Module:
    from models.faster_rcnn.model import build_faster_rcnn
    model = build_faster_rcnn(num_classes=num_classes + 1, pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_detr_model(weights_path: str, device: str, num_classes: int = 10) -> nn.Module:
    from models.detr.model import build_detr
    model = build_detr(num_classes=num_classes, pretrained_coco=True)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_model(model_type: str, weights_path: str, device: str, num_classes: int = 10) -> nn.Module:
    """Load model based on type (always custom-trained, not torchvision COCO-pretrained)."""
    if model_type == "faster_rcnn":
        return load_faster_rcnn_model(weights_path, device, num_classes)
    elif model_type == "yolo":
        return load_yolo_model(weights_path, device, num_classes)
    elif model_type == "detr":
        return load_detr_model(weights_path, device, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def preprocess_image(image_path: Path, input_size: int = 640) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float]]:
    """Load and preprocess an image. Returns tensor, original size, and scale factors."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((input_size, input_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    scale_x = input_size / float(orig_w)
    scale_y = input_size / float(orig_h)
    return img_tensor, (orig_w, orig_h), (scale_x, scale_y)


def _box_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a2x, a2y = ax + aw, ay + ah
    b2x, b2y = bx + bw, by + bh
    xi1, yi1 = max(ax, bx), max(ay, by)
    xi2, yi2 = min(a2x, b2x), min(a2y, b2y)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list[list[float]], scores: list[float], labels: list[int], iou_threshold: float) -> tuple[list[list[float]], list[float], list[int]]:
    """Per-class NMS dùng torchvision.ops.batched_nms (C++ kernel, nhanh hơn nhiều
    so với Python loop khi có hàng nghìn detection / ảnh).
    """
    if not boxes:
        return [], [], []

    from torchvision.ops import batched_nms

    boxes_xywh = torch.tensor(boxes, dtype=torch.float32)
    boxes_xyxy = torch.stack(
        [
            boxes_xywh[:, 0],
            boxes_xywh[:, 1],
            boxes_xywh[:, 0] + boxes_xywh[:, 2],
            boxes_xywh[:, 1] + boxes_xywh[:, 3],
        ],
        dim=1,
    )
    scores_t = torch.tensor(scores, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    keep_idx = batched_nms(boxes_xyxy, scores_t, labels_t, iou_threshold)
    keep_idx_list = keep_idx.tolist()

    keep_boxes = [boxes[i] for i in keep_idx_list]
    keep_scores = [float(scores[i]) for i in keep_idx_list]
    keep_labels = [int(labels[i]) for i in keep_idx_list]
    return keep_boxes, keep_scores, keep_labels


def predict_yolo(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: str,
    conf_threshold: float = 0.5,
    input_size: int = 640,
    orig_size: tuple[int, int] | None = None,
    scale: tuple[float, float] | None = None,
    nms_iou: float = 0.45,
) -> list[dict]:
    """Run prediction with YOLO model. Output bbox in original-image-space (xywh)."""
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device))

    if orig_size is None or scale is None:
        raise ValueError("orig_size and scale are required for YOLO decoding")

    orig_w, orig_h = orig_size
    scale_x, scale_y = scale

    boxes: list[list[float]] = []
    scores: list[float] = []
    labels: list[int] = []
    if not isinstance(outputs, list):
        return []

    # YOLO head output: list[(B, A*(5+C), H, W)] at multiple scales.
    # Vectorise decode để tránh Python loop O(num_cells) — quan trọng khi conf
    # threshold thấp (e.g. 0.05) sinh ra hàng nghìn detection / ảnh.
    for scale_idx, out in enumerate(outputs):
        bsz, ch, h, w = out.shape
        if bsz != 1:
            continue
        num_classes = ch // 3 - 5
        pred = out.view(1, 3, 5 + num_classes, h, w)[0]  # (A, 5+C, H, W)

        obj = pred[:, 4, :, :].sigmoid()
        cls = pred[:, 5:, :, :].sigmoid()
        # YOLOv5 parameterization (đồng bộ với YOLOLoss):
        #   pred_xy = 2*sigmoid(t) - 0.5  ∈ [-0.5, 1.5]
        #   pred_wh = (2*sigmoid(t))^2    ∈ [0, 4]
        raw_box = pred[:, :4, :, :].sigmoid()
        box_xy = 2.0 * raw_box[:, :2, :, :] - 0.5
        anchors = torch.tensor(DEFAULT_YOLO_ANCHORS[scale_idx], dtype=out.dtype, device=out.device)
        box_wh = (2.0 * raw_box[:, 2:, :, :]).pow(2)
        box_wh[:, 0, :, :] = box_wh[:, 0, :, :] * anchors[:, 0].view(3, 1, 1)
        box_wh[:, 1, :, :] = box_wh[:, 1, :, :] * anchors[:, 1].view(3, 1, 1)
        stride_x = input_size / float(w)
        stride_y = input_size / float(h)

        scores_full = obj.unsqueeze(1) * cls  # (A, C, H, W)
        best_score, best_cls = scores_full.max(dim=1)  # (A, H, W)
        keep = best_score > conf_threshold
        if not keep.any():
            continue
        a_idx, y_idx, x_idx = torch.where(keep)

        # Vectorise box decoding sang numpy 1-D arrays.
        sel_score = best_score[a_idx, y_idx, x_idx].cpu().numpy()
        sel_cls = best_cls[a_idx, y_idx, x_idx].cpu().numpy()
        sel_tx = box_xy[a_idx, 0, y_idx, x_idx].cpu().numpy()
        sel_ty = box_xy[a_idx, 1, y_idx, x_idx].cpu().numpy()
        sel_tw = box_wh[a_idx, 0, y_idx, x_idx].cpu().numpy()
        sel_th = box_wh[a_idx, 1, y_idx, x_idx].cpu().numpy()
        cx_grid = x_idx.cpu().numpy()
        cy_grid = y_idx.cpu().numpy()

        cx = (cx_grid + sel_tx) * stride_x
        cy = (cy_grid + sel_ty) * stride_y
        bw = sel_tw * input_size
        bh = sel_th * input_size

        x1_in = cx - bw / 2.0
        y1_in = cy - bh / 2.0
        x_arr = x1_in / scale_x
        y_arr = y1_in / scale_y
        w_arr = bw / scale_x
        h_arr = bh / scale_y

        x_arr = np.clip(x_arr, 0.0, float(orig_w))
        y_arr = np.clip(y_arr, 0.0, float(orig_h))
        w_arr = np.minimum(w_arr, float(orig_w) - x_arr)
        h_arr = np.minimum(h_arr, float(orig_h) - y_arr)
        valid = (w_arr > 0) & (h_arr > 0)

        for k in np.where(valid)[0]:
            boxes.append([float(x_arr[k]), float(y_arr[k]), float(w_arr[k]), float(h_arr[k])])
            scores.append(float(sel_score[k]))
            labels.append(int(sel_cls[k]))

    boxes, scores, labels = _nms(boxes, scores, labels, nms_iou)
    return [
        {"bbox": box, "score": float(score), "label": int(label)}
        for box, score, label in zip(boxes, scores, labels)
    ]


def predict_faster_rcnn(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: str,
    conf_threshold: float = 0.5,
    orig_size: tuple[int, int] | None = None,
    scale: tuple[float, float] | None = None,
) -> list[dict]:
    """Run prediction with Faster R-CNN model. Output bbox in original-image-space."""
    model.eval()
    with torch.no_grad():
        output = model([image_tensor.to(device)])
    
    if orig_size is None or scale is None:
        raise ValueError("orig_size and scale are required for Faster R-CNN decoding")

    orig_w, orig_h = orig_size
    scale_x, scale_y = scale

    predictions = []
    boxes = output[0]["boxes"].cpu().numpy()
    scores = output[0]["scores"].cpu().numpy()
    labels = output[0]["labels"].cpu().numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        if score > conf_threshold:
            x1_in, y1_in, x2_in, y2_in = box
            x1 = x1_in / scale_x
            y1 = y1_in / scale_y
            w = (x2_in - x1_in) / scale_x
            h = (y2_in - y1_in) / scale_y
            x1 = max(0.0, min(float(x1), float(orig_w)))
            y1 = max(0.0, min(float(y1), float(orig_h)))
            w = min(float(w), float(orig_w) - x1)
            h = min(float(h), float(orig_h) - y1)
            # Faster R-CNN: label 1..N (0 = background) → đổi về 0..N-1.
            cls_idx = int(label) - 1
            if cls_idx < 0:
                continue
            predictions.append({
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
                "label": cls_idx,
            })
    return predictions


def predict_detr(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: str,
    conf_threshold: float = 0.5,
    input_size: int = 640,
    orig_size: tuple[int, int] | None = None,
    scale: tuple[float, float] | None = None,
) -> list[dict]:
    """Run prediction with custom DETR model. Returns bbox in original-image-space (xywh)."""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))

    if orig_size is None or scale is None:
        raise ValueError("orig_size and scale are required for DETR decoding")

    orig_w, orig_h = orig_size
    scale_x, scale_y = scale

    predictions = []
    probas = output["pred_logits"].softmax(-1)[0]  # (Q, C+1)
    boxes = output["pred_boxes"][0].cpu().numpy()  # (Q, 4) normalized cxcywh

    for logits, box in zip(probas, boxes):
        max_logit, label = logits[:-1].max(0)
        if max_logit.item() > conf_threshold:
            cx, cy, w, h = box
            x1_in = (cx - w / 2) * input_size
            y1_in = (cy - h / 2) * input_size
            bw_in = w * input_size
            bh_in = h * input_size
            x = x1_in / scale_x
            y = y1_in / scale_y
            w_out = bw_in / scale_x
            h_out = bh_in / scale_y
            x = max(0.0, min(float(x), float(orig_w)))
            y = max(0.0, min(float(y), float(orig_h)))
            w_out = min(float(w_out), float(orig_w) - x)
            h_out = min(float(h_out), float(orig_h) - y)
            if w_out <= 0 or h_out <= 0:
                continue
            predictions.append({
                "bbox": [float(x), float(y), float(w_out), float(h_out)],
                "score": float(max_logit),
                "label": int(label),
            })
    return predictions


def predict_model(
    model: nn.Module,
    model_type: str,
    image_tensor: torch.Tensor,
    device: str,
    conf_threshold: float,
    input_size: int,
    orig_size: tuple[int, int],
    scale: tuple[float, float],
    nms_iou: float,
) -> list[dict]:
    """Run prediction based on model type."""
    if model_type == "yolo":
        return predict_yolo(
            model,
            image_tensor,
            device,
            conf_threshold,
            input_size,
            orig_size,
            scale,
            nms_iou,
        )
    # Note: predict_yolo defaults input_size=640. Đồng bộ với preprocess_image (resize 640).
    elif model_type == "faster_rcnn":
        return predict_faster_rcnn(model, image_tensor, device, conf_threshold, orig_size, scale)
    elif model_type == "detr":
        return predict_detr(model, image_tensor, device, conf_threshold, input_size, orig_size, scale)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _load_coco_mappings(annotations_path: Path) -> tuple[dict[str, int], dict[int, int]]:
    data = json.loads(annotations_path.read_text())
    image_id_by_file = {
        str(img.get("file_name")): int(img.get("id"))
        for img in data.get("images", [])
        if img.get("file_name") is not None and img.get("id") is not None
    }

    categories = data.get("categories", [])
    cat_name_to_id = {
        str(cat.get("name")): int(cat.get("id"))
        for cat in categories
        if cat.get("name") is not None and cat.get("id") is not None
    }

    classes_path = annotations_path.parent / "classes.txt"
    if classes_path.exists():
        class_names = [
            line.strip() for line in classes_path.read_text().splitlines() if line.strip()
        ]
    else:
        class_names = [
            str(cat.get("name")) for cat in categories if cat.get("name") is not None
        ]

    class_id_to_cat_id = {
        idx: cat_name_to_id[name]
        for idx, name in enumerate(class_names)
        if name in cat_name_to_id
    }

    return image_id_by_file, class_id_to_cat_id


def main():
    args = parse_args()
    
    print(f"Loading {args.model} model from {args.weights}...")
    model = load_model(args.model, args.weights, args.device, args.num_classes)
    model.eval()
    
    image_dir = Path(args.image_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in image_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images. Generating predictions...")
    
    image_id_by_file: dict[str, int] = {}
    class_id_to_cat_id: dict[int, int] = {}
    if args.annotations:
        annotations_path = Path(args.annotations)
        if annotations_path.exists():
            image_id_by_file, class_id_to_cat_id = _load_coco_mappings(annotations_path)
        else:
            print(f"[Warning] Annotations file not found: {annotations_path}")

    all_predictions: list[dict] = []
    for img_path in image_files:
        image_tensor, orig_size, scale = preprocess_image(img_path, args.input_size)
        predictions = predict_model(
            model,
            args.model,
            image_tensor,
            args.device,
            args.conf_threshold,
            args.input_size,
            orig_size,
            scale,
            args.nms_iou,
        )

        image_id = image_id_by_file.get(img_path.name)
        if image_id is None:
            if img_path.stem.isdigit():
                image_id = int(img_path.stem)
            else:
                image_id = img_path.stem

        for pred in predictions:
            label = int(pred.pop("label"))
            category_id = class_id_to_cat_id.get(label, label)
            pred["image_id"] = image_id
            pred["category_id"] = category_id
            all_predictions.append(pred)

        print(f"  {img_path.name}: {len(predictions)} detections")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\nPredictions saved to {output_path}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total detections: {len(all_predictions)}")


if __name__ == "__main__":
    main()
