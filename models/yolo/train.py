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
from datetime import datetime

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
from models.utils.coco_dataset import get_coco_dataloaders, get_class_names, CocoDetection, collate_fn
from models.utils.losses import YOLOLoss, DEFAULT_YOLO_ANCHORS
from models.utils.box_ops import cxcywh_to_xyxy, box_iou


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
    p.add_argument("--num_workers", type=int, default=train_config.get("num_workers", 4),
                    help=f"Số workers cho DataLoader (mặc định: {train_config.get('num_workers', 4)})")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Thiết bị chạy (cpu hoặc cuda)")
    p.add_argument("--base_dir", type=str, default="models/yolo",
                    help="Thư mục gốc lưu checkpoints và logs (mặc định: models/yolo)")
    p.add_argument("--resume", type=str, default=None,
                    help="Đường dẫn checkpoint để resume training")
    p.add_argument("--augment", action="store_true",
                    help="Sử dụng data augmentation (RandomHorizontalFlip, ColorJitter, RandomRotation)")
    p.add_argument("--output", type=str, default=None,
                    help="Đường dẫn lưu checkpoint cuối cùng")
    p.add_argument("--conf_threshold", type=float, default=infer_config.get("conf_threshold", 0.25),
                    help=f"Confidence threshold cho evaluation (mặc định: {infer_config.get('conf_threshold', 0.25)})")
    p.add_argument("--iou_threshold", type=float, default=infer_config.get("iou_threshold", 0.5),
                    help=f"IoU threshold cho mAP (mặc định: {infer_config.get('iou_threshold', 0.5)})")
    return p.parse_args()


def build_dataloaders(args: argparse.Namespace, classes: list[str]):
    """Build dataloaders từ dataset."""
    from torch.utils.data import DataLoader

    print(f"[YOLO] Loading datasets from {args.data_root}/...")

    train_dataset = CocoDetection(
        img_folder=Path(args.data_root) / "images" / "train",
        ann_file=Path(args.data_root) / "annotations" / "train.json",
        classes=classes,
        img_size=args.img_size,
        augment=args.augment,
    )

    val_dataset = CocoDetection(
        img_folder=Path(args.data_root) / "images" / "val",
        ann_file=Path(args.data_root) / "annotations" / "val.json",
        classes=classes,
        img_size=args.img_size,
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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
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
    anchors = torch.tensor(DEFAULT_YOLO_ANCHORS, dtype=torch.float32)
    batch_size = predictions[0].shape[0]

    for batch_idx in range(batch_size):
        batch_boxes: list[dict] = []

        for scale_idx, pred in enumerate(predictions):
            _, _, h, w = pred.shape
            stride = strides[scale_idx]

            # Reshape: (B, anchors * (5+C), H, W) -> (B, anchors, 5+C, H, W)
            num_anchors = 3
            pred = pred.view(batch_size, num_anchors, 5 + num_classes, h, w)

            # Lấy prediction cho batch hiện tại
            pred = pred[batch_idx]  # (anchors, 5+C, H, W)

            for anchor_idx in range(num_anchors):
                anchor_pred = pred[anchor_idx]  # (5+C, H, W)
                anchor_w = float(anchors[scale_idx, anchor_idx, 0])
                anchor_h = float(anchors[scale_idx, anchor_idx, 1])

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
    
    # Lưu tất cả predictions và targets theo class
    # predictions[class_id] = list of (confidence, is_TP)
    class_preds: dict[int, list[tuple[float, bool]]] = {i: [] for i in range(num_classes)}
    class_gts: dict[int, int] = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            outputs = model(images)

            # Decode predictions từ YOLO output
            predictions = decode_predictions(
                outputs,
                conf_threshold=conf_threshold,
                num_classes=num_classes,
                img_size=img_size,
            )

            for img_idx, img_preds in enumerate(predictions):
                # Lấy image_id từ target (là tensor [img_id])
                target = targets[img_idx]
                img_id_tensor = target.get("image_id", torch.tensor([0]))
                img_id = img_id_tensor.item() if torch.is_tensor(img_id_tensor) else img_id_tensor
                if isinstance(img_id, (list, tuple)):
                    img_id = img_id[0]

                # Apply NMS
                img_preds = nms(img_preds, iou_threshold)

                # Lấy ground truths cho ảnh này
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

                # Đếm GTs theo class
                for label in img_gt_labels:
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    if 0 <= label < num_classes:
                        class_gts[label] += 1

                # Match predictions với GTs để xác định TP/FP
                for pred in img_preds:
                    cls_id = pred["class"]
                    if 0 <= cls_id < num_classes:
                        pred_box = pred["bbox"]

                        # Tìm GT cùng class có IoU cao nhất
                        best_iou = 0.0
                        matched = False

                        for gt_idx, (gt_box, gt_label) in enumerate(zip(img_gt_boxes, img_gt_labels)):
                            if isinstance(gt_label, torch.Tensor):
                                gt_label = gt_label.item()
                            if gt_label != cls_id:
                                continue

                            # Tính IoU
                            x1 = max(pred_box[0], gt_box[0])
                            y1 = max(pred_box[1], gt_box[1])
                            x2 = min(pred_box[2], gt_box[2])
                            y2 = min(pred_box[3], gt_box[3])

                            inter_w = max(0, x2 - x1)
                            inter_h = max(0, y2 - y1)
                            inter_area = inter_w * inter_h

                            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                            union_area = pred_area + gt_area - inter_area

                            iou = inter_area / (union_area + 1e-10)

                            if iou > best_iou:
                                best_iou = iou

                        is_tp = best_iou >= iou_threshold
                        class_preds[cls_id].append((pred["score"], is_tp))

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


def main():
    args = parse_args()
    device = torch.device(args.device)

    classes = get_class_names(Path(args.data_root) / "annotations")
    num_classes = len(classes)

    checkpoint_dir = Path(args.base_dir) / "checkpoints"
    log_dir = Path(args.base_dir) / "logs"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[YOLO] Checkpoints: {checkpoint_dir}")
    print(f"[YOLO] Logs: {log_dir}")
    
    writer = SummaryWriter(log_dir=log_dir)
    
    train_loader, val_loader = build_dataloaders(args, classes)

    print(f"[YOLO] Building model with {num_classes} classes: {classes}")
    model = build_yolo(num_classes=num_classes)
    model.to(device)

    criterion = YOLOLoss(num_classes=num_classes, num_anchors=3, image_size=args.img_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_loss = 0.0  # mAP cao hơn = tốt hơn, nên khởi tạo = 0

    if args.resume:
        print(f"[YOLO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"[YOLO] Starting training for {args.epochs} epochs...")
    print(f"[YOLO] Device: {device}")
    print(f"[YOLO] Batch size: {args.batch_size}")
    print(f"[YOLO] Learning rate: {args.lr}")
    print(f"[YOLO] Data augmentation: {args.augment}")
    print(f"[YOLO] Conf threshold: {args.conf_threshold}")
    print(f"[YOLO] IoU threshold: {args.iou_threshold}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], epoch)

        # Evaluate mAP
        print(f"[YOLO] Evaluating mAP@{args.iou_threshold}...")
        metrics = evaluate_model(
            model, val_loader, device,
            num_classes=num_classes,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            img_size=args.img_size,
        )
        mAP = metrics["mAP"]
        writer.add_scalar("epoch/mAP", mAP, epoch)

        print(f"[YOLO] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, mAP={mAP:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        if mAP > best_val_loss:  # Use mAP for model selection
            best_val_loss = mAP
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "classes": classes,
                "conf_threshold": args.conf_threshold,
                "iou_threshold": args.iou_threshold,
            }, best_checkpoint_path)
            print(f"[YOLO] Saved best model (mAP={mAP:.4f}) to {best_checkpoint_path}")

        if args.output:
            final_path = Path(args.output)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "classes": classes,
                "conf_threshold": args.conf_threshold,
                "iou_threshold": args.iou_threshold,
            }, final_path)
            print(f"[YOLO] Saved final model to {final_path}")

    writer.close()
    print("[YOLO] Training completed!")


if __name__ == "__main__":
    main()
