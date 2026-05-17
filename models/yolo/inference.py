"""Inference script cho YOLO object detection.

Sử dụng config.yaml để lấy thresholds.
Nếu không tìm thấy config, dùng defaults: conf=0.25, iou=0.45

Usage:
    python models/yolo/inference.py --checkpoint models/yolo/checkpoints/best_model.pt --img data/images/test/image.jpg
    python models/yolo/inference.py --checkpoint best_model.pt --img image.jpg --conf 0.5 --iou 0.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from models.yolo.model import build_yolo
from models.utils.box_ops import box_iou
from models.yolo.train import decode_predictions as train_decode_predictions
from models.yolo.train import nms as train_nms
from models.yolo.train import parse_anchor_config


def load_config(config_path: str | None = None) -> dict:
    """Load config from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def non_max_suppression(
    predictions: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_classes: int = 10,
) -> list[list[dict]]:
    """Apply NMS và lọc theo confidence threshold.
    
    Args:
        predictions: Tensor shape (batch, num_anchors * (5 + num_classes), num_predictions)
                     hoặc đã được reshape phù hợp.
        conf_threshold: Confidence threshold để lọc predictions.
        iou_threshold: IoU threshold cho NMS.
        num_classes: Số lớp.
    
    Returns:
        List of list of dicts, mỗi dict chứa box, score, class.
    """
    # Giả sử predictions đã được xử lý thành: [batch][num_boxes, (x1,y1,x2,y2,conf,class)]
    batch_size = predictions.shape[0]
    results = []

    for i in range(batch_size):
        pred = predictions[i]  # (num_boxes, 6) = (x1,y1,x2,y2,conf,class)
        
        # Lọc theo conf threshold
        mask = pred[:, 4] >= conf_threshold
        pred = pred[mask]

        if pred.shape[0] == 0:
            results.append([])
            continue

        # Sắp xếp theo confidence
        sort_idx = pred[:, 4].argsort(descending=True)
        pred = pred[sort_idx]

        # NMS per class
        keep_boxes = []
        classes = pred[:, 5].unique()

        for cls in classes:
            cls_mask = pred[:, 5] == cls
            cls_pred = pred[cls_mask]

            while cls_pred.shape[0] > 0:
                # Giữ box có conf cao nhất
                keep_boxes.append(cls_pred[0])
                
                if cls_pred.shape[0] == 1:
                    break

                # Tính IoU với các boxes còn lại
                ious = box_iou(
                    cls_pred[0:1, :4], 
                    cls_pred[1:, :4]
                ).squeeze(0)

                # Giữ các boxes có IoU < iou_threshold
                keep_mask = ious < iou_threshold
                cls_pred = cls_pred[1:][keep_mask]

        results.append(keep_boxes)

    return results


def preprocess_image(
    img_path: str,
    img_size: int = 640,
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int], float, int, int]:
    """Preprocess ảnh cho inference.
    
    Returns:
        Tensor đã normalize, ảnh gốc, original shape, scale, pad_x, pad_y.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    # Resize
    scale = img_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Pad
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    img_padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    img_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized

    # Normalize và convert sang tensor
    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0

    return img_tensor, img, (orig_h, orig_w), scale, pad_x, pad_y


def postprocess_predictions(
    outputs: list[torch.Tensor],
    img_size: int,
    orig_shape: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float,
    iou_threshold: float,
    num_classes: int,
    anchors: list[list[tuple[float, float]]] | None = None,
    num_anchors: int = 3,
) -> list[dict]:
    """Postprocess model outputs thành boxes.
    
    Cần implement theo output format của model.
    """
    if not isinstance(outputs, list):
        outputs = [outputs]

    decoded = train_decode_predictions(
        outputs,
        conf_threshold=conf_threshold,
        num_classes=num_classes,
        img_size=img_size,
        anchors=anchors,
    )
    if not decoded:
        return []

    preds = decoded[0]
    preds = train_nms(preds, iou_threshold=iou_threshold)

    orig_h, orig_w = orig_shape
    results: list[dict] = []
    for pred in preds:
        x1, y1, x2, y2 = pred["bbox"]
        # Reverse centered letterbox transform.
        x1 = (x1 - float(pad_x)) / scale
        y1 = (y1 - float(pad_y)) / scale
        x2 = (x2 - float(pad_x)) / scale
        y2 = (y2 - float(pad_y)) / scale

        x1 = float(np.clip(x1, 0.0, float(orig_w)))
        y1 = float(np.clip(y1, 0.0, float(orig_h)))
        x2 = float(np.clip(x2, 0.0, float(orig_w)))
        y2 = float(np.clip(y2, 0.0, float(orig_h)))
        if x2 <= x1 or y2 <= y1:
            continue

        results.append(
            {
                "bbox": [x1, y1, x2, y2],
                "score": float(pred["score"]),
                "class": int(pred["class"]),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO inference script.")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Đường dẫn checkpoint")
    p.add_argument("--img", type=str, required=True,
                    help="Đường dẫn ảnh input")
    p.add_argument("--img_size", type=int, default=None,
                    help="Kích thước ảnh (mặc định: lấy từ config)")
    p.add_argument("--conf", type=float, default=None,
                    help=f"Confidence threshold (mặc định: từ config)")
    p.add_argument("--iou", type=float, default=None,
                    help=f"IoU threshold cho NMS (mặc định: từ config)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Thiết bị (cpu hoặc cuda)")
    p.add_argument("--output", type=str, default=None,
                    help="Đường dẫn lưu ảnh kết quả")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Load config
    config = load_config()
    model_config = config.get("model", {})
    infer_config = config.get("inference", {})
    
    img_size = args.img_size or model_config.get("image_size", 640)
    conf_threshold = args.conf or infer_config.get("conf_threshold", 0.25)
    iou_threshold = args.iou or infer_config.get("iou_threshold", 0.45)
    num_classes = model_config.get("num_classes", 10)
    num_anchors = int(model_config.get("num_anchors", 3))
    classes = model_config.get("classes", [f"class_{i}" for i in range(num_classes)])
    
    print(f"[YOLO] Config: img_size={img_size}, conf={conf_threshold}, iou={iou_threshold}")
    print(f"[YOLO] Classes: {classes}")

    # Load checkpoint
    print(f"[YOLO] Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device)
    anchors = parse_anchor_config(checkpoint.get("anchors", model_config.get("anchors")), num_anchors=num_anchors)
    
    # Build model
    model = build_yolo(num_classes=num_classes, num_anchors=num_anchors)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocess image
    print(f"[YOLO] Processing: {args.img}")
    img_tensor, orig_img, orig_shape, scale, pad_x, pad_y = preprocess_image(args.img, img_size)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dim

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    print(f"[YOLO] Raw outputs shape: {[o.shape for o in outputs]}")

    results = postprocess_predictions(
        outputs=outputs,
        img_size=img_size,
        orig_shape=orig_shape,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        num_classes=num_classes,
        anchors=anchors,
        num_anchors=num_anchors,
    )

    print(f"[YOLO] Inference completed. {len(results)} detections.")

    # Draw boxes on original image.
    vis_img = orig_img.copy()
    for det in results:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id = det["class"]
        score = det["score"]
        label = classes[cls_id] if 0 <= cls_id < len(classes) else f"class_{cls_id}"

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (67, 142, 219), 2)
        cv2.putText(
            vis_img,
            f"{label} {score:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Save visualization (placeholder)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"[YOLO] Output saved to {output_path}")
    else:
        print("[YOLO] No --output provided, visualization was not saved.")


if __name__ == "__main__":
    main()
