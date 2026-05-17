"""Training script cho YOLO object detection.

Dataset structure expected:
    data/
    ├── annotations/
    │   ├── train.json      # COCO format
    │   ├── val.json
    │   ├── test.json
    │   └── classes.txt     # Danh sách class names
    └── images/
        ├── train/
        ├── val/
        └── test/

Usage:
    python models/yolo/train.py --data_root data --epochs 50 --batch_size 8

Checkpoints và logs được lưu tại:
    models/yolo/checkpoints/best_model.pt
    models/yolo/logs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
import torchvision
import torchvision.ops
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.yolo.model import build_yolo
from models.utils.coco_dataset import get_class_names, CocoDetection, collate_fn
from models.utils.losses import YOLOLoss, DEFAULT_YOLO_ANCHORS
from models.utils.box_ops import box_iou


def load_config(config_path: str | None = None) -> dict:
    """Load config from YAML file. Falls back to defaults if not found."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def parse_anchor_config(
    anchor_config: Any,
    num_anchors: int = 3,
) -> list[list[tuple[float, float]]]:
    """Parse anchors từ config YAML về format YOLOLoss cần dùng."""
    if anchor_config is None:
        return DEFAULT_YOLO_ANCHORS

    if not isinstance(anchor_config, list) or len(anchor_config) == 0:
        raise ValueError("[YOLO] model.anchors phải là list 3 scale, mỗi scale gồm 3 (w,h).")

    anchors_out: list[list[tuple[float, float]]] = []
    for scale_anchors in anchor_config:
        if not isinstance(scale_anchors, list):
            raise ValueError("[YOLO] model.anchors không đúng format list[list[...]].")
        scale_out: list[tuple[float, float]] = []
        for pair in scale_anchors:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("[YOLO] Mỗi anchor phải có đúng 2 giá trị (w, h).")
            w = float(pair[0])
            h = float(pair[1])
            if w <= 0 or h <= 0:
                raise ValueError("[YOLO] Anchor width/height phải > 0.")
            scale_out.append((w, h))
        anchors_out.append(scale_out)

    if any(len(s) != num_anchors for s in anchors_out):
        raise ValueError(
            f"[YOLO] Mỗi scale phải có đúng {num_anchors} anchors, got {[len(s) for s in anchors_out]}"
        )
    return anchors_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO object detection model.")
    p.add_argument("--config", type=str, default=None,
                    help="Đường dẫn tới config.yaml (mặc định: models/yolo/config.yaml)")
    
    # Pre-parse to get config path
    temp_args = p.parse_known_args()[0]
    config = load_config(temp_args.config)

    model_config = config.get("model", {})
    train_config = config.get("training", {})
    infer_config = config.get("inference", {})

    p.add_argument("--data_root", type=str, default="data",
                    help="Đường dẫn tới thư mục data/ (mặc định: data)")
    p.add_argument("--img_size", type=int, default=model_config.get("image_size", 640),
                    help=f"Kích thước ảnh input (mặc định: {model_config.get('image_size', 640)})")
    p.add_argument("--batch_size", type=int, default=train_config.get("batch_size", 8),
                    help=f"Batch size (mặc định: {train_config.get('batch_size', 8)})")
    p.add_argument("--epochs", type=int, default=train_config.get("epochs", 50),
                    help=f"Số epochs (mặc định: {train_config.get('epochs', 50)})")
    p.add_argument("--lr", type=float, default=train_config.get("lr", 0.001),
                    help=f"Learning rate (mặc định: {train_config.get('lr', 0.001)})")
    p.add_argument("--weight_decay", type=float, default=train_config.get("weight_decay", 0.0001),
                    help=f"Weight decay (mặc định: {train_config.get('weight_decay', 0.0001)})")
    p.add_argument("--warmup_epochs", type=int, default=train_config.get("warmup_epochs", 3),
                    help=f"Số epoch warmup LR (mặc định: {train_config.get('warmup_epochs', 3)})")
    p.add_argument("--min_lr_ratio", type=float, default=train_config.get("min_lr_ratio", 0.05),
                    help=f"LR tối thiểu = lr * min_lr_ratio (mặc định: {train_config.get('min_lr_ratio', 0.05)})")
    p.add_argument("--grad_clip_norm", type=float, default=train_config.get("grad_clip_norm", 10.0),
                    help=f"Clip grad norm (mặc định: {train_config.get('grad_clip_norm', 10.0)})")
    p.add_argument("--num_workers", type=int, default=train_config.get("num_workers", 4),
                    help=f"Số workers cho DataLoader (mặc định: {train_config.get('num_workers', 4)})")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Thiết bị chạy (cpu hoặc cuda)")
    p.add_argument("--base_dir", type=str, default="models/yolo",
                    help="Thư mục gốc lưu checkpoints và logs (mặc định: models/yolo)")
    p.add_argument("--resume", type=str, default=None,
                    help="Đường dẫn checkpoint để resume training")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction,
                    default=train_config.get("augment_enabled", True),
                    help=f"Bật/tắt data augmentation (mặc định: {train_config.get('augment_enabled', True)})")
    p.add_argument("--disable_letterbox", action="store_true",
                    help="Tắt letterbox, quay về resize méo 640x640 (không khuyến nghị)")
    p.add_argument("--use_coco_eval", action=argparse.BooleanOptionalAction,
                    default=train_config.get("use_coco_eval", True),
                    help=f"Bật/tắt COCO-style evaluation (mặc định: {train_config.get('use_coco_eval', True)})")
    p.add_argument("--output", type=str, default=None,
                    help="Đường dẫn lưu checkpoint cuối cùng")
    p.add_argument("--conf_threshold", type=float, default=infer_config.get("conf_threshold", 0.25),
                    help=f"Confidence threshold cho evaluation (mặc định: {infer_config.get('conf_threshold', 0.25)})")
    p.add_argument("--iou_threshold", type=float, default=infer_config.get("iou_threshold", 0.45),
                    help=f"IoU threshold cho mAP (mặc định: {infer_config.get('iou_threshold', 0.45)})")
    return p.parse_args()


def build_dataloaders(
    args: argparse.Namespace,
    classes: list[str],
    augment_cfg: dict[str, Any] | None = None,
):
    """Build dataloaders từ dataset."""
    from torch.utils.data import DataLoader

    print(f"[YOLO] Loading datasets from {args.data_root}/...")

    train_dataset = CocoDetection(
        img_folder=Path(args.data_root) / "images" / "train",
        ann_file=Path(args.data_root) / "annotations" / "train.json",
        classes=classes,
        img_size=args.img_size,
        augment=args.augment,
        letterbox=not args.disable_letterbox,
        augment_cfg=augment_cfg,
    )

    val_dataset = CocoDetection(
        img_folder=Path(args.data_root) / "images" / "val",
        ann_file=Path(args.data_root) / "annotations" / "val.json",
        classes=classes,
        img_size=args.img_size,
        letterbox=not args.disable_letterbox,
        augment_cfg=augment_cfg,
    )

    print(f"[YOLO] Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    writer,
    grad_clip_norm: float = 10.0,
):
    """Train một epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        target_boxes = [t["boxes"].to(device) for t in targets]
        target_labels = [t["labels"].to(device) for t in targets]

        loss = criterion(outputs, target_boxes, target_labels)

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("train/loss", loss.item(), global_step)

    return total_loss / len(dataloader)


@torch.no_grad
def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc="Validating"):
        images = images.to(device)

        outputs = model(images)

        target_boxes = [t["boxes"].to(device) for t in targets]
        target_labels = [t["labels"].to(device) for t in targets]

        loss = criterion(outputs, target_boxes, target_labels)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision (AP) cho một class."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def decode_predictions(
    predictions: list[torch.Tensor],
    conf_threshold: float = 0.25,
    num_classes: int = 5,
    img_size: int = 640,
    anchors: list[list[tuple[float, float]]] | None = None,
) -> list[list[dict]]:
    """Decode YOLO predictions thành list of boxes.

    Args:
        predictions: List of 3 tensors từ YOLO model output.
            Mỗi tensor shape: (batch, num_anchors * (5 + num_classes), H, W)
        conf_threshold: Confidence threshold để lọc predictions
        num_classes: Số lượng classes

    Returns:
        List of lists, outer list = batch, inner list = predictions
        mỗi prediction: {"bbox": [x1, y1, x2, y2], "score": float, "class": int}
    """
    all_batch_predictions: list[list[dict]] = []

    # Strides cho 3 scale (P3, P4, P5) - phải khớp với neck output
    strides = [8, 16, 32]
    anchor_values = anchors if anchors is not None else DEFAULT_YOLO_ANCHORS
    anchors_tensor = torch.tensor(anchor_values, dtype=torch.float32)
    num_anchors = int(anchors_tensor.shape[1])
    batch_size = predictions[0].shape[0]

    for batch_idx in range(batch_size):
        batch_boxes: list[dict] = []

        for scale_idx, pred in enumerate(predictions):
            _, _, h, w = pred.shape
            stride = strides[scale_idx]

            # Reshape: (B, anchors * (5+C), H, W) -> (B, anchors, 5+C, H, W)
            pred = pred.view(batch_size, num_anchors, 5 + num_classes, h, w)

            # Lấy prediction cho batch hiện tại
            pred = pred[batch_idx]  # (anchors, 5+C, H, W)

            for anchor_idx in range(num_anchors):
                anchor_pred = pred[anchor_idx]  # (5+C, H, W)
                anchor_w = float(anchors_tensor[scale_idx, anchor_idx, 0])
                anchor_h = float(anchors_tensor[scale_idx, anchor_idx, 1])

                # Tách các thành phần
                tx = anchor_pred[0]   # (H, W)
                ty = anchor_pred[1]
                tw = anchor_pred[2]
                th = anchor_pred[3]
                obj = torch.sigmoid(anchor_pred[4])  # (H, W)
                cls = torch.sigmoid(anchor_pred[5:5+num_classes])  # (C, H, W)

                # YOLOv5 decode: bbox coordinates trong feature map
                # pred_xy = 2*sigmoid(tx) - 0.5 ∈ [-0.5, 1.5]
                # pred_wh = (2*sigmoid(t))^2 ∈ [0, 4]
                bx = 2 * torch.sigmoid(tx) - 0.5  # (H, W)
                by = 2 * torch.sigmoid(ty) - 0.5
                bw = ((2 * torch.sigmoid(tw)) ** 2) * anchor_w   # (H, W)
                bh = ((2 * torch.sigmoid(th)) ** 2) * anchor_h

                # Grid indices - đảm bảo nằm trên đúng device
                grid_device = pred.device
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(h, dtype=torch.float32, device=grid_device),
                    torch.arange(w, dtype=torch.float32, device=grid_device),
                    indexing='ij'
                )

                # Absolute coordinates trong feature map
                gx = (x_grid + bx) * stride
                gy = (y_grid + by) * stride
                # YOLOLoss optimize pred_wh as fraction of image size,
                # nên cần scale theo img_size (không phải stride).
                gw = bw * float(img_size)
                gh = bh * float(img_size)

                # Convert xywh -> xyxy (top-left, bottom-right)
                x1 = gx - gw / 2
                y1 = gy - gh / 2
                x2 = gx + gw / 2
                y2 = gy + gh / 2

                # Tính confidence = objectness * max_class_prob
                max_cls_prob, top_cls = cls.max(dim=0)  # (H, W)
                confidence = obj * max_cls_prob  # (H, W)

                # Lọc theo confidence threshold
                mask = confidence > conf_threshold
                if not mask.any():
                    continue

                x1_filtered = x1[mask].detach().cpu().numpy()
                y1_filtered = y1[mask].detach().cpu().numpy()
                x2_filtered = x2[mask].detach().cpu().numpy()
                y2_filtered = y2[mask].detach().cpu().numpy()
                scores = confidence[mask].detach().cpu().numpy()
                classes = top_cls[mask].detach().cpu().numpy()

                # Clamp về frame ảnh input để tránh box âm/vượt khung làm IoU sai.
                x1_filtered = np.clip(x1_filtered, 0.0, float(img_size))
                y1_filtered = np.clip(y1_filtered, 0.0, float(img_size))
                x2_filtered = np.clip(x2_filtered, 0.0, float(img_size))
                y2_filtered = np.clip(y2_filtered, 0.0, float(img_size))

                for i in range(len(scores)):
                    if x2_filtered[i] <= x1_filtered[i] or y2_filtered[i] <= y1_filtered[i]:
                        continue
                    batch_boxes.append({
                        "bbox": [float(x1_filtered[i]), float(y1_filtered[i]),
                                 float(x2_filtered[i]), float(y2_filtered[i])],
                        "score": float(scores[i]),
                        "class": int(classes[i]),
                    })

        all_batch_predictions.append(batch_boxes)

    return all_batch_predictions


def nms_single_class(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
) -> torch.Tensor:
    """Apply NMS cho predictions cùng một class (dùng torchvision).

    Args:
        boxes: (N, 4) tensor [x1, y1, x2, y2]
        scores: (N,) tensor confidence scores
        iou_threshold: IoU threshold cho suppression

    Returns:
        Indices của boxes được giữ lại
    """
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)
    return torchvision.ops.nms(boxes, scores, iou_threshold)


def nms(
    predictions: list[dict],
    iou_threshold: float = 0.45,
) -> list[dict]:
    """Apply NMS cho tất cả predictions.

    Args:
        predictions: List of dicts với keys: bbox [x1,y1,x2,y2], score, class
        iou_threshold: IoU threshold cho suppression

    Returns:
        Filtered list of predictions sau NMS
    """
    if len(predictions) == 0:
        return []

    # Extract to tensors
    boxes_list = [p["bbox"] for p in predictions]
    scores = torch.tensor([p["score"] for p in predictions])
    classes = torch.tensor([p["class"] for p in predictions])

    boxes = torch.tensor(boxes_list)
    if boxes.shape[-1] != 4:
        return predictions  # Fallback

    # Apply NMS per class
    keep_indices_all = []
    for cls in classes.unique().tolist():
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        keep = nms_single_class(cls_boxes, cls_scores, iou_threshold)
        # Map back to original indices
        cls_indices = torch.where(cls_mask)[0]
        keep_indices_all.extend(cls_indices[keep].tolist())

    return [predictions[i] for i in sorted(keep_indices_all)]


def evaluate_model(
    model,
    dataloader,
    device,
    num_classes: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    anchors: list[list[tuple[float, float]]] | None = None,
) -> dict[str, float]:
    """Evaluate model và tính mAP@IoU.
    
    Args:
        model: YOLO model
        dataloader: Validation dataloader
        device: Device để chạy
        num_classes: Số lượng classes
        conf_threshold: Confidence threshold để lọc predictions
        iou_threshold: IoU threshold cho NMS và tính mAP
        img_size: Kích thước ảnh input
    
    Returns:
        Dict chứa mAP và các metrics khác
    """
    model.eval()
    
    # predictions[class_id] = list of (confidence, is_true_positive)
    class_preds: dict[int, list[tuple[float, bool]]] = {i: [] for i in range(num_classes)}
    class_gts: dict[int, int] = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            predictions = decode_predictions(
                outputs,
                conf_threshold=conf_threshold,
                num_classes=num_classes,
                img_size=img_size,
                anchors=anchors,
            )

            for img_idx, img_preds in enumerate(predictions):
                target = targets[img_idx]
                img_preds = nms(img_preds, iou_threshold)

                img_gt_boxes = target.get("boxes", torch.tensor([]))
                img_gt_labels = target.get("labels", torch.tensor([]))

                if isinstance(img_gt_boxes, torch.Tensor):
                    img_gt_boxes = img_gt_boxes.cpu().tolist()
                if isinstance(img_gt_labels, torch.Tensor):
                    img_gt_labels = img_gt_labels.cpu().tolist()
                if not isinstance(img_gt_boxes, list):
                    img_gt_boxes = []
                if not isinstance(img_gt_labels, list):
                    img_gt_labels = []

                gt_by_cls = {i: [] for i in range(num_classes)}
                for gt_box, gt_label in zip(img_gt_boxes, img_gt_labels):
                    if isinstance(gt_label, torch.Tensor):
                        gt_label = gt_label.item()
                    gt_label = int(gt_label)
                    if 0 <= gt_label < num_classes:
                        class_gts[gt_label] += 1
                        gt_by_cls[gt_label].append([float(v) for v in gt_box])

                matched_by_cls: dict[int, list[bool]] = {
                    cls_id: [False] * len(gt_boxes_cls)
                    for cls_id, gt_boxes_cls in gt_by_cls.items()
                }

                img_preds = sorted(img_preds, key=lambda x: float(x["score"]), reverse=True)

                for pred in img_preds:
                    cls_id = int(pred["class"])
                    if cls_id < 0 or cls_id >= num_classes:
                        continue

                    pred_box = torch.tensor([pred["bbox"]], dtype=torch.float32)
                    gt_boxes_cls = gt_by_cls.get(cls_id, [])
                    matched_flags = matched_by_cls.get(cls_id, [])

                    best_iou = 0.0
                    best_gt_idx = -1
                    if gt_boxes_cls:
                        gt_tensor = torch.tensor(gt_boxes_cls, dtype=torch.float32)
                        ious = box_iou(pred_box, gt_tensor).squeeze(0).tolist()
                        for gt_idx, iou in enumerate(ious):
                            if gt_idx < len(matched_flags) and matched_flags[gt_idx]:
                                continue
                            if float(iou) > best_iou:
                                best_iou = float(iou)
                                best_gt_idx = gt_idx

                    is_tp = best_gt_idx >= 0 and best_iou >= iou_threshold
                    if is_tp:
                        matched_flags[best_gt_idx] = True
                    class_preds[cls_id].append((float(pred["score"]), bool(is_tp)))

    # Tính AP cho mỗi class
    aps = []
    for cls in range(num_classes):
        preds = class_preds[cls]
        num_gts = class_gts[cls]
        
        if num_gts == 0:
            aps.append(0.0)
            continue
        
        if len(preds) == 0:
            aps.append(0.0)
            continue

        # Sort predictions by confidence descending
        preds.sort(key=lambda x: x[0], reverse=True)
        
        tp = np.array([1 if p[1] else 0 for p in preds], dtype=np.float32)
        fp = np.array([0 if p[1] else 1 for p in preds], dtype=np.float32)
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Precision-Recall curve
        recall = tp_cumsum / num_gts
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        ap = compute_ap(recall, precision)
        aps.append(ap)

    # mAP@iou_threshold
    mAP = np.mean(aps) if aps else 0.0

    return {
        "mAP": mAP,
        "mAP@0.5": mAP,
        "APs": aps,
        "num_gts": sum(class_gts.values()),
    }


@torch.no_grad()
def evaluate_coco_map(
    model,
    dataloader,
    device,
    num_classes: int,
    class_names: list[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    anchors: list[list[tuple[float, float]]] | None = None,
) -> dict[str, float]:
    """Đánh giá mAP chuẩn COCO trên val dataloader."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        print(f"[YOLO][Warn] pycocotools không khả dụng, bỏ qua COCO eval: {exc}")
        return {"mAP_50_95": 0.0, "mAP_50": 0.0, "available": 0.0}

    model.eval()

    images_payload: list[dict[str, Any]] = []
    annotations_payload: list[dict[str, Any]] = []
    detections_payload: list[dict[str, Any]] = []
    seen_images: set[int] = set()
    ann_id = 1

    for images, targets in tqdm(dataloader, desc="COCO Evaluating"):
        images = images.to(device)
        outputs = model(images)
        predictions = decode_predictions(
            outputs,
            conf_threshold=conf_threshold,
            num_classes=num_classes,
            img_size=img_size,
            anchors=anchors,
        )

        for img_idx, img_preds in enumerate(predictions):
            target = targets[img_idx]
            img_id_tensor = target.get("image_id", torch.tensor([0]))
            if isinstance(img_id_tensor, torch.Tensor):
                img_id = int(img_id_tensor.view(-1)[0].item())
            elif isinstance(img_id_tensor, (list, tuple)):
                img_id = int(img_id_tensor[0])
            else:
                img_id = int(img_id_tensor)

            if img_id not in seen_images:
                images_payload.append({"id": img_id, "width": img_size, "height": img_size})
                seen_images.add(img_id)

            img_preds = nms(img_preds, iou_threshold)
            for pred in img_preds:
                x1, y1, x2, y2 = [float(v) for v in pred["bbox"]]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                cls_id = int(pred["class"])
                if cls_id < 0 or cls_id >= num_classes:
                    continue
                detections_payload.append(
                    {
                        "image_id": img_id,
                        "category_id": cls_id + 1,
                        "bbox": [x1, y1, w, h],
                        "score": float(pred["score"]),
                    }
                )

            gt_boxes = target.get("boxes", torch.zeros((0, 4)))
            gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.int64))
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu()
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu()

            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                cls_id = int(gt_label.item()) if torch.is_tensor(gt_label) else int(gt_label)
                if cls_id < 0 or cls_id >= num_classes:
                    continue
                x1 = float(gt_box[0])
                y1 = float(gt_box[1])
                x2 = float(gt_box[2])
                y2 = float(gt_box[3])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                annotations_payload.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id + 1,
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    categories_payload = [
        {"id": idx + 1, "name": class_names[idx], "supercategory": "animal"}
        for idx in range(num_classes)
    ]

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images_payload,
        "annotations": annotations_payload,
        "categories": categories_payload,
    }
    coco_gt.createIndex()

    if not annotations_payload:
        return {"mAP_50_95": 0.0, "mAP_50": 0.0, "available": 1.0}

    if not detections_payload:
        return {"mAP_50_95": 0.0, "mAP_50": 0.0, "available": 1.0}

    coco_dt = coco_gt.loadRes(detections_payload)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = sorted(list(seen_images))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP_50_95": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "available": 1.0,
    }


class WarmupCosineScheduler:
    """Linear warmup + cosine decay scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        total_epochs: int,
        warmup_epochs: int = 3,
        min_lr_ratio: float = 0.05,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr_ratio = float(min_lr_ratio)
        self.current_epoch = 0

    def _compute_lr(self, epoch: int) -> float:
        min_lr = self.base_lr * self.min_lr_ratio
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            progress = float(epoch + 1) / float(self.warmup_epochs)
            return max(min_lr, self.base_lr * progress)

        cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)
        if cosine_epochs == 1:
            return min_lr
        progress = float(epoch - self.warmup_epochs) / float(cosine_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr + (self.base_lr - min_lr) * cosine

    def step(self) -> float:
        lr = self._compute_lr(self.current_epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_epoch += 1
        return lr

    def get_last_lr(self) -> list[float]:
        return [float(pg["lr"]) for pg in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {"current_epoch": self.current_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.current_epoch = int(state_dict.get("current_epoch", 0))


def main():
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    model_config = config.get("model", {})
    train_config = config.get("training", {})

    classes = get_class_names(Path(args.data_root) / "annotations")
    num_classes = len(classes)
    num_anchors = int(model_config.get("num_anchors", 3))
    anchors = parse_anchor_config(model_config.get("anchors"), num_anchors=num_anchors)
    augment_cfg = train_config.get("augment", {})
    if not isinstance(augment_cfg, dict):
        augment_cfg = {}

    checkpoint_dir = Path(args.base_dir) / "checkpoints"
    log_dir = Path(args.base_dir) / "logs"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[YOLO] Checkpoints: {checkpoint_dir}")
    print(f"[YOLO] Logs: {log_dir}")
    
    writer = SummaryWriter(log_dir=log_dir)
    
    train_loader, val_loader = build_dataloaders(args, classes, augment_cfg=augment_cfg)

    print(f"[YOLO] Building model with {num_classes} classes: {classes}")
    model = build_yolo(
        num_classes=num_classes,
        base_channels=int(model_config.get("base_channels", 32)),
        num_anchors=num_anchors,
    )
    model.to(device)

    criterion = YOLOLoss(
        num_classes=num_classes,
        num_anchors=num_anchors,
        image_size=args.img_size,
        anchors=anchors,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        base_lr=args.lr,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
    )

    start_epoch = 0
    best_val_loss = float("inf")
    best_map_score = float("-inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"[YOLO] Resume checkpoint không tồn tại: {resume_path}")
        print(f"[YOLO] Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_map_score = float(checkpoint.get("best_map_score", float("-inf")))
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception:
                # Checkpoint cũ có scheduler khác type, fallback fast-forward.
                scheduler.current_epoch = start_epoch
        else:
            scheduler.current_epoch = start_epoch
        if scheduler.current_epoch < start_epoch:
            scheduler.current_epoch = start_epoch

        print(f"[YOLO] Resumed at epoch {start_epoch}, "
              f"best val_loss so far={best_val_loss:.4f}, "
              f"best_map so far={best_map_score:.4f}, "
              f"current lr={optimizer.param_groups[0]['lr']:.6f}")

    print(f"[YOLO] Starting training for {args.epochs} epochs...")
    print(f"[YOLO] Device: {device}")
    print(f"[YOLO] Batch size: {args.batch_size}")
    print(f"[YOLO] Learning rate: {args.lr}")
    print(f"[YOLO] Warmup epochs: {args.warmup_epochs}, min_lr_ratio: {args.min_lr_ratio}")
    print(f"[YOLO] Grad clip norm: {args.grad_clip_norm}")
    print(f"[YOLO] Data augmentation: {args.augment}")
    print(f"[YOLO] Letterbox: {not args.disable_letterbox}")
    print(f"[YOLO] COCO eval: {args.use_coco_eval}")
    print(f"[YOLO] Conf threshold: {args.conf_threshold}")
    print(f"[YOLO] IoU threshold: {args.iou_threshold}")

    for epoch in range(start_epoch, args.epochs):
        lr_now = scheduler.step()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_loss = validate(model, val_loader, criterion, device)

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/lr", lr_now, epoch)

        # Evaluate mAP
        print(f"[YOLO] Evaluating mAP@{args.iou_threshold}...")
        metrics = evaluate_model(
            model, val_loader, device,
            num_classes=num_classes,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            img_size=args.img_size,
            anchors=anchors,
        )
        map_custom = float(metrics["mAP"])
        writer.add_scalar("epoch/mAP_custom", map_custom, epoch)

        coco_metrics = {"available": 0.0, "mAP_50_95": 0.0, "mAP_50": 0.0}
        if args.use_coco_eval:
            coco_metrics = evaluate_coco_map(
                model=model,
                dataloader=val_loader,
                device=device,
                num_classes=num_classes,
                class_names=classes,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                img_size=args.img_size,
                anchors=anchors,
            )
            if coco_metrics.get("available", 0.0) > 0:
                writer.add_scalar("epoch/mAP_coco_50_95", coco_metrics["mAP_50_95"], epoch)
                writer.add_scalar("epoch/mAP_coco_50", coco_metrics["mAP_50"], epoch)

        selection_map = (
            float(coco_metrics["mAP_50_95"])
            if coco_metrics.get("available", 0.0) > 0
            else map_custom
        )

        print(
            f"[YOLO] Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, mAP_custom={map_custom:.4f}, "
            f"mAP_select={selection_map:.4f}, lr={lr_now:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if selection_map > best_map_score:
            best_map_score = selection_map
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_map_score": best_map_score,
                "classes": classes,
                "anchors": anchors,
                "conf_threshold": args.conf_threshold,
                "iou_threshold": args.iou_threshold,
                "metrics": {
                    "custom_map": map_custom,
                    "coco_map_50_95": float(coco_metrics.get("mAP_50_95", 0.0)),
                    "coco_map_50": float(coco_metrics.get("mAP_50", 0.0)),
                },
            }, best_checkpoint_path)
            print(f"[YOLO] Saved best model (mAP_select={best_map_score:.4f}) to {best_checkpoint_path}")

        if args.output:
            final_path = Path(args.output)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_map_score": best_map_score,
                "classes": classes,
                "anchors": anchors,
                "conf_threshold": args.conf_threshold,
                "iou_threshold": args.iou_threshold,
            }, final_path)
            print(f"[YOLO] Saved final model to {final_path}")

    writer.close()
    print("[YOLO] Training completed!")


if __name__ == "__main__":
    main()
