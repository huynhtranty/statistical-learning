"""Training script cho Faster R-CNN object detection.

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
    python models/faster_rcnn/train.py --data_root data --epochs 50 --batch_size 4

Checkpoints và logs được lưu tại:
    models/faster_rcnn/checkpoints/best_model.pt
    models/faster_rcnn/logs/
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

from models.faster_rcnn.model import build_faster_rcnn
from models.utils.coco_dataset import get_coco_dataloaders, get_class_names, CocoDetection, collate_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Faster R-CNN object detection model.")
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
    p.add_argument("--num_workers", type=int, default=4,
                    help="Số workers cho DataLoader (mặc định: 4)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Thiết bị chạy (cpu hoặc cuda)")
    p.add_argument("--base_dir", type=str, default="models/faster_rcnn",
                    help="Thư mục gốc lưu checkpoints và logs (mặc định: models/faster_rcnn)")
    p.add_argument("--resume", type=str, default=None,
                    help="Đường dẫn checkpoint để resume training")
    p.add_argument("--augment", action="store_true",
                    help="Sử dụng data augmentation (RandomHorizontalFlip, ColorJitter, RandomRotation)")
    p.add_argument("--output", type=str, default=None,
                    help="Đường dẫn lưu checkpoint cuối cùng")
    return p.parse_args()


class FasterRCNNDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper để chuyển đổi CocoDetection output sang format của Faster R-CNN."""

    def __init__(self, coco_dataset: CocoDetection, device: str = "cpu"):
        self.dataset = coco_dataset
        self.device = device

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        return image, target


def build_dataloaders(args: argparse.Namespace, classes: list[str]):
    """Build dataloaders từ dataset."""
    from torch.utils.data import DataLoader

    print(f"[Faster R-CNN] Loading datasets from {args.data_root}/...")

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

    print(f"[Faster R-CNN] Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

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


def train_one_epoch(model, dataloader, optimizer, device, epoch, writer):
    """Train một epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, targets) in enumerate(pbar):
        # images is a stacked tensor [B, C, H, W]; list() converts to list of tensors
        images = list(images.to(device))

        target_dicts = []
        for target in targets:
            # Dataset trả label 0..N-1; Faster R-CNN giữ class 0 cho background.
            # → cộng 1 để label trở thành 1..N (background = 0).
            target_dict = {
                "boxes": target["boxes"].to(device),
                "labels": (target["labels"] + 1).to(device),
            }
            if "masks" in target:
                target_dict["masks"] = target["masks"].to(device)
            target_dicts.append(target_dict)

        loss_dict = model(images, target_dicts)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix({"loss": f"{losses.item():.4f}"})

        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("train/total_loss", losses.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc="Validating"):
        # images is already a stacked tensor from collate_fn, targets is a list
        images = images.to(device)

        target_dicts = []
        for target in targets:
            target_dict = {
                "boxes": target["boxes"].to(device),
                "labels": (target["labels"] + 1).to(device),
            }
            target_dicts.append(target_dict)

        loss_dict = model(images, target_dicts)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()

    return total_loss / len(dataloader)


def main():
    args = parse_args()
    device = torch.device(args.device)

    classes = get_class_names(Path(args.data_root) / "annotations")
    num_classes = len(classes)

    # Faster R-CNN convention: num_classes bao gồm background (label 0).
    # → truyền (num_classes + 1). Dataset cần map label về 1..N (xem build_dataloaders).
    print(f"[Faster R-CNN] Building model with {num_classes} foreground classes + 1 background: {classes}")
    model = build_faster_rcnn(num_classes=num_classes + 1)
    model.to(device)

    checkpoint_dir = Path(args.base_dir) / "checkpoints"
    log_dir = Path(args.base_dir) / "logs"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Faster R-CNN] Checkpoints: {checkpoint_dir}")
    print(f"[Faster R-CNN] Logs: {log_dir}")

    train_loader, val_loader = build_dataloaders(args, classes)

    writer = SummaryWriter(log_dir=log_dir)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"[Faster R-CNN] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"[Faster R-CNN] Starting training for {args.epochs} epochs...")
    print(f"[Faster R-CNN] Device: {device}")
    print(f"[Faster R-CNN] Batch size: {args.batch_size}")
    print(f"[Faster R-CNN] Learning rate: {args.lr}")
    print(f"[Faster R-CNN] Data augmentation: {args.augment}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, writer
        )
        val_loss = validate(model, val_loader, device)

        scheduler.step()

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], epoch)

        print(f"[Faster R-CNN] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "classes": classes,
            }, best_checkpoint_path)
            print(f"[Faster R-CNN] Saved best model to {best_checkpoint_path}")

        if args.output:
            final_path = Path(args.output)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "classes": classes,
            }, final_path)
            print(f"[Faster R-CNN] Saved final model to {final_path}")

    writer.close()
    print("[Faster R-CNN] Training completed!")


if __name__ == "__main__":
    main()
