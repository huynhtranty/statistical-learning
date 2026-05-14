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
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.yolo.model import build_yolo
from models.utils.coco_dataset import get_coco_dataloaders, get_class_names, CocoDetection, collate_fn
from models.utils.losses import YOLOLoss
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
    # predictions[class_id] = list of (confidence, TP/FP)
    class_preds: dict[int, list[tuple[float, bool]]] = {i: [] for i in range(num_classes)}
    class_gts: dict[int, int] = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            # TODO: Decode outputs thành boxes
            # Hiện tại cần implement decode logic dựa trên model output format
            
            # Đếm ground truths
            for target in targets:
                labels = target["labels"]
                for label in labels:
                    class_gts[label.item() if hasattr(label, 'item') else label] += 1

    # Tính AP cho mỗi class
    aps = []
    for cls in range(num_classes):
        preds = class_preds[cls]
        num_gts = class_gts[cls]
        
        if num_gts == 0 or len(preds) == 0:
            aps.append(0.0)
            continue

        preds.sort(key=lambda x: x[0], reverse=True)
        
        tp = np.array([1 if p[1] else 0 for p in preds], dtype=np.float32)
        fp = np.array([0 if p[1] else 1 for p in preds], dtype=np.float32)
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / num_gts
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
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

    criterion = YOLOLoss(num_classes=num_classes, num_anchors=3, image_size=args.img_size)
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
