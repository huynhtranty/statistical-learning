#!/usr/bin/env python3
"""Visualize ground truth bboxes lấy trực tiếp từ DataLoader.

Mục đích: xác nhận pipeline dataset → collate_fn KHÔNG làm hỏng bbox.
Nếu GT vẽ ra đúng object → dataset OK; nếu lệch → bug ở dataset/resize/augment.

Usage:
    python scripts/visualize_dataset.py --data_root data --split train \\
        --output /tmp/gt_vis --num 12 --augment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.utils.coco_dataset import CocoDetection, collate_fn, get_class_names


def draw(img_np: np.ndarray, boxes: torch.Tensor, labels: torch.Tensor, classes: list[str]) -> np.ndarray:
    """Vẽ bbox xyxy (pixel) trên ảnh RGB (numpy)."""
    img = img_np.copy()
    palette = [
        (235, 122, 67), (67, 142, 219), (130, 235, 149), (235, 176, 33),
        (198, 120, 255), (129, 236, 236), (255, 200, 150), (180, 130, 90),
        (220, 220, 120), (100, 200, 200),
    ]
    for box, lab in zip(boxes.tolist(), labels.tolist()):
        x1, y1, x2, y2 = map(int, box)
        color = palette[int(lab) % len(palette)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = classes[int(lab)] if int(lab) < len(classes) else f"cls_{lab}"
        cv2.putText(img, name, (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--output", default="/tmp/gt_vis")
    p.add_argument("--num", type=int, default=12)
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--augment", action="store_true")
    args = p.parse_args()

    classes = get_class_names(Path(args.data_root) / "annotations")
    ds = CocoDetection(
        img_folder=Path(args.data_root) / "images" / args.split,
        ann_file=Path(args.data_root) / "annotations" / f"{args.split}.json",
        classes=classes,
        img_size=args.img_size,
        augment=args.augment,
    )
    print(f"[INFO] Dataset {args.split}: {len(ds)} images, classes={classes}")

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for imgs, targets in dl:
        # imgs is tensor (B, 3, H, W) in [0,1]
        for i in range(imgs.shape[0]):
            if saved >= args.num:
                break
            img_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            boxes = targets[i]["boxes"]
            labels = targets[i]["labels"]

            print(f"[{saved+1:02d}] boxes={boxes.shape[0]}, "
                  f"box xyxy mean=[{boxes.float().mean(0).tolist() if boxes.numel() else 'empty'}], "
                  f"labels={labels.tolist()}")

            drawn = draw(img_np, boxes, labels, classes)
            out_file = out_dir / f"gt_{saved:03d}.jpg"
            cv2.imwrite(str(out_file), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
            saved += 1
        if saved >= args.num:
            break

    print(f"\n[DONE] Saved {saved} GT visualizations to {out_dir}")
    print("→ Mở xem xem bbox có nằm đúng object không.")
    print("  Nếu sai: dataset/resize có bug. Nếu đúng: dataset OK, vấn đề ở loss/model.")


if __name__ == "__main__":
    main()
