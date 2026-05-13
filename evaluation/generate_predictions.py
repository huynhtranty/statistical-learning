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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions for evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=["faster_rcnn", "yolo", "detr"],
                        help="Model type")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for predictions")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run inference on")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Confidence threshold for predictions")
    parser.add_argument("--input-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="Number of classes in the model")
    return parser.parse_args()


def load_yolo_model(weights_path: str, device: str, num_classes: int = 10) -> nn.Module:
    """Load YOLO model."""
    from models.yolo.model import build_yolo
    model = build_yolo(num_classes=num_classes)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    return model.to(device)


def load_faster_rcnn_model(weights_path: str, device: str) -> nn.Module:
    """Load Faster R-CNN model."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_detr_model(weights_path: str, device: str) -> nn.Module:
    """Load DETR model."""
    from torchvision.models.detection import detr_resnet50, DetrResNet50_Weights
    model = detr_resnet50(weights=DetrResNet50_Weights.DEFAULT)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_model(model_type: str, weights_path: str, device: str, num_classes: int = 10) -> nn.Module:
    """Load model based on type."""
    if model_type == "faster_rcnn":
        return load_faster_rcnn_model(weights_path, device)
    elif model_type == "yolo":
        return load_yolo_model(weights_path, device, num_classes)
    elif model_type == "detr":
        return load_detr_model(weights_path, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def preprocess_image(image_path: Path, input_size: int = 640) -> torch.Tensor:
    """Load and preprocess an image."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_size, input_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return img_tensor


def predict_yolo(model: nn.Module, image_tensor: torch.Tensor, device: str,
                 conf_threshold: float = 0.5) -> list[dict]:
    """Run prediction with YOLO model."""
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device))

    predictions = []
    if not isinstance(outputs, list):
        return predictions

    # YOLO head output: list[(B, A*(5+C), H, W)] at multiple scales.
    for out in outputs:
        bsz, ch, h, w = out.shape
        if bsz != 1:
            continue
        num_classes = ch // 3 - 5
        pred = out.view(1, 3, 5 + num_classes, h, w)[0]  # (A, 5+C, H, W)

        obj = pred[:, 4, :, :].sigmoid()
        cls = pred[:, 5:, :, :].sigmoid()
        box = pred[:, :4, :, :].sigmoid()
        stride = 640 / float(h)

        for a in range(pred.shape[0]):
            scores_map, labels_map = (obj[a].unsqueeze(0) * cls[a]).max(dim=0)
            ys, xs = torch.where(scores_map > conf_threshold)
            for y, x in zip(ys.tolist(), xs.tolist()):
                score = float(scores_map[y, x])
                label = int(labels_map[y, x])

                tx, ty, tw, th = box[a, :, y, x]
                cx = (x + float(tx)) * stride
                cy = (y + float(ty)) * stride
                bw = max(1.0, float(tw) * 640.0)
                bh = max(1.0, float(th) * 640.0)

                predictions.append({
                    "bbox": [float(cx - bw / 2.0), float(cy - bh / 2.0), float(bw), float(bh)],
                    "score": score,
                    "class": label,
                })
    return predictions


def predict_faster_rcnn(model: nn.Module, image_tensor: torch.Tensor, device: str,
                        conf_threshold: float = 0.5) -> list[dict]:
    """Run prediction with Faster R-CNN model."""
    model.eval()
    with torch.no_grad():
        output = model([image_tensor.to(device)])
    
    predictions = []
    boxes = output[0]["boxes"].cpu().numpy()
    scores = output[0]["scores"].cpu().numpy()
    labels = output[0]["labels"].cpu().numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        if score > conf_threshold:
            x1, y1, x2, y2 = box
            predictions.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
                "class": int(label)
            })
    return predictions


def predict_detr(model: nn.Module, image_tensor: torch.Tensor, device: str,
                  conf_threshold: float = 0.5) -> list[dict]:
    """Run prediction with DETR model."""
    model.eval()
    with torch.no_grad():
        output = model([image_tensor.to(device)])
    
    predictions = []
    probas = output["pred_logits"].softmax(-1)
    boxes = output["pred_boxes"]
    
    for logits, box in zip(probas, boxes):
        max_logit, label = logits[:-1].max(0)
        if max_logit > conf_threshold:
            x, y, w, h = box.cpu().numpy()
            predictions.append({
                "bbox": [float(x - w/2), float(y - h/2), float(w), float(h)],
                "score": float(max_logit),
                "class": int(label)
            })
    return predictions


def predict_model(model: nn.Module, model_type: str, image_tensor: torch.Tensor,
                   device: str, conf_threshold: float) -> list[dict]:
    """Run prediction based on model type."""
    if model_type == "yolo":
        return predict_yolo(model, image_tensor, device, conf_threshold)
    elif model_type == "faster_rcnn":
        return predict_faster_rcnn(model, image_tensor, device, conf_threshold)
    elif model_type == "detr":
        return predict_detr(model, image_tensor, device, conf_threshold)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
    
    all_predictions = []
    for img_path in image_files:
        image_tensor = preprocess_image(img_path, args.input_size)
        predictions = predict_model(model, args.model, image_tensor, 
                                     args.device, args.conf_threshold)
        
        all_predictions.append({
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "predictions": predictions
        })
        
        print(f"  {img_path.name}: {len(predictions)} detections")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\nPredictions saved to {output_path}")
    print(f"Total images processed: {len(all_predictions)}")
    print(f"Total detections: {sum(len(p['predictions']) for p in all_predictions)}")


if __name__ == "__main__":
    main()
