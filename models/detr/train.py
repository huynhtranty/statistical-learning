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
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.detr.model import build_detr
from models.utils.coco_dataset import get_coco_dataloaders, get_class_names, CocoDetection, collate_fn
from models.utils.losses import SetCriterion
from models.detr.matcher import HungarianMatcher


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
    p.add_argument("--checkpoint_dir", type=str, default="models/detr/checkpoints",
                    help="Thư mục lưu checkpoints")
    p.add_argument("--log_dir", type=str, default="models/detr/logs",
                    help="Thư mục lưu logs")
    p.add_argument("--resume", type=str, default=None,
                    help="Đường dẫn checkpoint để resume training")
    p.add_argument("--aux_loss", action="store_true", default=True,
                    help="Sử dụng auxiliary loss từ mỗi decoder layer")
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


def prepare_targets(targets: list[dict], device: torch.device) -> list[dict]:
    """Chuẩn bị targets cho DETR training."""
    prepared = []
    for target in targets:
        prepared.append({
            "labels": target["labels"].to(device),
            "boxes": target["boxes"].to(device),
        })
    return prepared


def train_one_epoch(model, dataloader, matcher, criterion, optimizer, device, epoch, writer, aux_loss=True):
    """Train một epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = prepare_targets(targets, device)

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
def validate(model, dataloader, matcher, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        targets = prepare_targets(targets, device)

        outputs = model(images)
        indices = matcher(outputs, targets)
        loss_dict = criterion(outputs, targets, indices)

        weight_dict = criterion.weight_dict
        losses = sum(weight_dict.get(k, 0) * v for k, v in loss_dict.items())

        total_loss += losses.item()

    return total_loss / len(dataloader)


def main():
    args = parse_args()
    device = torch.device(args.device)

    classes = get_class_names(Path(args.data_root) / "annotations")
    num_classes = len(classes)

    print(f"[DETR] Building model with {num_classes} classes: {classes}")
    print(f"[DETR] Using {args.num_queries} object queries")
    print(f"[DETR] Auxiliary loss: {args.aux_loss}")

    model = build_detr(num_classes=num_classes, num_queries=args.num_queries)
    model.to(device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    train_loader, val_loader = build_dataloaders(args, classes)

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
        print(f"[DETR] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"[DETR] Starting training for {args.epochs} epochs...")
    print(f"[DETR] Device: {device}")
    print(f"[DETR] Batch size: {args.batch_size}")
    print(f"[DETR] Learning rate: {args.lr}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, matcher, criterion, optimizer, device, epoch, writer, args.aux_loss
        )
        val_loss = validate(model, val_loader, matcher, criterion, device)

        scheduler.step()

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], epoch)

        print(f"[DETR] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "classes": classes,
                "num_queries": args.num_queries,
            }, checkpoint_path)
            print(f"[DETR] Saved best model to {checkpoint_path}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": val_loss,
                "classes": classes,
                "num_queries": args.num_queries,
            }, checkpoint_path)

    writer.close()
    print("[DETR] Training completed!")


if __name__ == "__main__":
    main()
