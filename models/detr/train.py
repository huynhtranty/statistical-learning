"""Training script cho DETR (DEtection TRansformer).

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
    python models/detr/train.py --data_root data --epochs 50 --batch_size 4

Checkpoints và logs được lưu tại:
    models/detr/checkpoints/best_model.pt
    models/detr/logs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision.ops
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.detr.model import build_detr
from models.utils.coco_dataset import get_coco_dataloaders, get_class_names, CocoDetection, collate_fn
from models.utils.losses import SetCriterion
from models.detr.matcher import HungarianMatcher
from models.utils.detection_metrics import evaluate_detection_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DETR object detection model.")
    p.add_argument("--data_root", type=str, default="data",
                    help="Đường dẫn tới thư mục data/ (mặc định: data)")
    p.add_argument("--img_size", type=int, default=640,
                    help="Kích thước ảnh input (mặc định: 640)")
    p.add_argument("--batch_size", type=int, default=4,
                    help="Batch size (mặc định: 4)")
    p.add_argument("--epochs", type=int, default=50,
                    help="Số epochs (mặc định: 50)")
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate (mặc định: 1e-4)")
    p.add_argument("--num_queries", type=int, default=100,
                    help="Số object queries (mặc định: 100)")
    p.add_argument("--num_workers", type=int, default=4,
                    help="Số workers cho DataLoader (mặc định: 4)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Thiết bị chạy (cpu hoặc cuda)")
    p.add_argument("--base_dir", type=str, default="models/detr",
                    help="Thư mục gốc lưu checkpoints và logs (mặc định: models/detr)")
    p.add_argument("--resume", type=str, default=None,
                    help="Đường dẫn checkpoint để resume training")
    p.add_argument("--augment", action="store_true",
                    help="Sử dụng data augmentation (RandomHorizontalFlip, ColorJitter, RandomRotation)")
    p.add_argument("--aux_loss", action=argparse.BooleanOptionalAction, default=True,
                    help="Bật/tắt auxiliary loss từ mỗi decoder layer")
    p.add_argument("--output", type=str, default=None,
                    help="Đường dẫn lưu checkpoint cuối cùng")
    p.add_argument("--conf_threshold", type=float, default=0.25,
                    help="Confidence threshold cho evaluation mAP")
    p.add_argument("--iou_threshold", type=float, default=0.5,
                    help="IoU threshold cho mAP")
    return p.parse_args()


def build_dataloaders(args: argparse.Namespace, classes: list[str]):
    """Build dataloaders từ dataset."""
    from torch.utils.data import DataLoader

    print(f"[DETR] Loading datasets from {args.data_root}/...")

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

    print(f"[DETR] Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

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


def prepare_targets(targets: list[dict], device: torch.device, img_size: int) -> list[dict]:
    """Chuyển targets từ dataset format (xyxy pixel) sang DETR format
    (cxcywh chuẩn hoá [0,1]) và move sang device."""
    prepared = []
    for target in targets:
        boxes = target["boxes"].to(device).float()
        if boxes.numel() > 0:
            x1, y1, x2, y2 = boxes.unbind(-1)
            cx = (x1 + x2) * 0.5 / img_size
            cy = (y1 + y2) * 0.5 / img_size
            bw = (x2 - x1) / img_size
            bh = (y2 - y1) / img_size
            boxes = torch.stack((cx, cy, bw, bh), dim=-1).clamp(0.0, 1.0)
        prepared.append({
            "labels": target["labels"].to(device),
            "boxes": boxes,
        })
    return prepared


def train_one_epoch(model, dataloader, matcher, criterion, optimizer, device, epoch, writer, img_size, aux_loss=True):
    """Train một epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = prepare_targets(targets, device, img_size)

        outputs = model(images)

        indices = matcher(outputs, targets)
        loss_dict = criterion(outputs, targets, indices)

        weight_dict = criterion.weight_dict
        losses = sum(weight_dict.get(k, 0) * v for k, v in loss_dict.items())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("train/loss", losses.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)

    return total_loss / len(dataloader)


@torch.no_grad
def validate(model, dataloader, matcher, criterion, device, img_size):
    """Validate model."""
    model.eval()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        targets = prepare_targets(targets, device, img_size)

        outputs = model(images)
        indices = matcher(outputs, targets)
        loss_dict = criterion(outputs, targets, indices)

        weight_dict = criterion.weight_dict
        losses = sum(weight_dict.get(k, 0) * v for k, v in loss_dict.items())

        total_loss += losses.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes: int, img_size: int, conf_threshold: float, iou_threshold: float):
    model.eval()
    all_predictions: list[list[dict]] = []
    all_targets: list[dict] = []

    for images, targets in tqdm(dataloader, desc="Evaluating mAP"):
        images = images.to(device)
        outputs = model(images)

        pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
        pred_boxes = outputs["pred_boxes"]    # (B, Q, 4) normalized cxcywh
        probas = pred_logits.softmax(-1)

        for b_idx, target in enumerate(targets):
            boxes_norm = pred_boxes[b_idx].detach().cpu()
            probs = probas[b_idx].detach().cpu()

            scores, labels = probs[:, :-1].max(dim=-1)
            keep = scores >= conf_threshold
            boxes_norm = boxes_norm[keep]
            scores = scores[keep]
            labels = labels[keep]

            pred_list: list[dict] = []
            if boxes_norm.numel() > 0:
                cx = boxes_norm[:, 0] * img_size
                cy = boxes_norm[:, 1] * img_size
                bw = boxes_norm[:, 2] * img_size
                bh = boxes_norm[:, 3] * img_size
                x1 = (cx - bw / 2.0).clamp(0.0, float(img_size))
                y1 = (cy - bh / 2.0).clamp(0.0, float(img_size))
                x2 = (cx + bw / 2.0).clamp(0.0, float(img_size))
                y2 = (cy + bh / 2.0).clamp(0.0, float(img_size))
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

                valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
                boxes_xyxy = boxes_xyxy[valid]
                scores = scores[valid]
                labels = labels[valid]

                if boxes_xyxy.numel() > 0:
                    keep_idx = torchvision.ops.batched_nms(boxes_xyxy, scores, labels, iou_threshold)
                    for idx in keep_idx.tolist():
                        lb = int(labels[idx].item())
                        if lb < 0 or lb >= num_classes:
                            continue
                        bx = boxes_xyxy[idx].tolist()
                        pred_list.append({
                            "bbox": [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])],
                            "score": float(scores[idx].item()),
                            "class": lb,
                        })

            all_predictions.append(pred_list)
            all_targets.append({
                "boxes": target["boxes"].detach().cpu().tolist(),
                "labels": target["labels"].detach().cpu().tolist(),
            })

    return evaluate_detection_metrics(all_predictions, all_targets, num_classes, iou_threshold=iou_threshold)


def main():
    args = parse_args()
    device = torch.device(args.device)

    classes = get_class_names(Path(args.data_root) / "annotations")
    num_classes = len(classes)

    print(f"[DETR] Building model with {num_classes} classes: {classes}")
    print(f"[DETR] Using {args.num_queries} object queries")
    print(f"[DETR] Auxiliary loss: {args.aux_loss}")

    # Mặc định dùng HuggingFace facebook/detr-resnet-50 COCO-pretrained,
    # chỉ thay class head cho num_classes của ta — đồng bộ với Faster R-CNN
    # cũng COCO-pretrained và YOLO ImageNet-pretrained backbone.
    model = build_detr(
        num_classes=num_classes,
        num_queries=args.num_queries,
        pretrained_coco=True,
    )
    model.to(device)

    checkpoint_dir = Path(args.base_dir) / "checkpoints"
    log_dir = Path(args.base_dir) / "logs"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DETR] Checkpoints: {checkpoint_dir}")
    print(f"[DETR] Logs: {log_dir}")

    train_loader, val_loader = build_dataloaders(args, classes)

    writer = SummaryWriter(log_dir=log_dir)

    matcher = HungarianMatcher()
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2},
        aux_loss=args.aux_loss,
    )
    criterion.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"[DETR] Resume checkpoint không tồn tại: {resume_path}")
        print(f"[DETR] Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            for _ in range(checkpoint.get("epoch", -1) + 1):
                scheduler.step()
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"[DETR] Resumed at epoch {start_epoch}, "
              f"best val_loss so far={best_val_loss:.4f}, "
              f"current lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"[DETR] Starting training for {args.epochs} epochs...")
    print(f"[DETR] Device: {device}")
    print(f"[DETR] Batch size: {args.batch_size}")
    print(f"[DETR] Learning rate: {args.lr}")
    print(f"[DETR] Data augmentation: {args.augment}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, matcher, criterion, optimizer, device, epoch, writer,
            args.img_size, args.aux_loss,
        )
        val_loss = validate(model, val_loader, matcher, criterion, device, args.img_size)
        metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            num_classes=num_classes,
            img_size=args.img_size,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
        )

        scheduler.step()

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/mAP", metrics["mAP"], epoch)
        writer.add_scalar("epoch/mean_iou", metrics["mean_iou"], epoch)
        writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], epoch)

        print(
            f"[DETR] Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, mAP={metrics['mAP']:.4f}, "
            f"mean_iou={metrics['mean_iou']:.4f}, lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "classes": classes,
                "num_queries": args.num_queries,
            }, best_checkpoint_path)
            print(f"[DETR] Saved best model (val_loss={best_val_loss:.4f}) to {best_checkpoint_path}")

        if args.output:
            final_path = Path(args.output)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "classes": classes,
                "num_queries": args.num_queries,
            }, final_path)
            print(f"[DETR] Saved final model to {final_path}")

    writer.close()
    print("[DETR] Training completed!")


if __name__ == "__main__":
    main()
