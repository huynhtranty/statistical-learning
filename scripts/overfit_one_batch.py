#!/usr/bin/env python3
"""Overfit YOLO model trên 1 batch nhỏ — sanity check kinh điển.

Nếu model + loss đúng, sau ~500 step trên CÙNG 1 batch, loss phải gần 0
và inference trên chính batch đó phải dự đoán gần đúng bboxes.

Nếu loss KHÔNG giảm xuống gần 0 → bug ở loss/model/pipeline.
Nếu loss giảm nhưng inference vẫn random → bug ở inference decode.

Usage:
    python scripts/overfit_one_batch.py --data_root data --steps 500 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.utils.coco_dataset import CocoDetection, collate_fn, get_class_names
from models.utils.losses import YOLOLoss
from models.yolo.model import build_yolo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", default=None, help="optional save checkpoint sau overfit")
    args = p.parse_args()

    device = torch.device(args.device)
    classes = get_class_names(Path(args.data_root) / "annotations")
    num_classes = len(classes)

    ds = CocoDetection(
        img_folder=Path(args.data_root) / "images" / "train",
        ann_file=Path(args.data_root) / "annotations" / "train.json",
        classes=classes,
        img_size=args.img_size,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Lấy 1 batch DUY NHẤT và lặp nó qua tất cả steps
    imgs, targets = next(iter(dl))
    imgs = imgs.to(device)
    target_boxes = [t["boxes"].to(device) for t in targets]
    target_labels = [t["labels"].to(device) for t in targets]

    print(f"[INFO] Overfitting on batch of {imgs.shape[0]} images, "
          f"{sum(len(b) for b in target_boxes)} objects total")
    print(f"[INFO] Classes per image: {[t.tolist() for t in target_labels]}")

    model = build_yolo(num_classes=num_classes, pretrained_backbone=True).to(device)
    model.train()
    crit = YOLOLoss(num_classes=num_classes, num_anchors=3, image_size=args.img_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    print(f"\n{'step':>5} | {'loss':>8} | {'obj_max':>8} | {'cls_max':>8} | {'wh_mean':>10}")
    print("-" * 60)
    for step in range(args.steps):
        opt.zero_grad()
        outs = model(imgs)
        loss = crit(outs, target_boxes, target_labels)
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == args.steps - 1:
            with torch.no_grad():
                # Inspect P5 output
                o = outs[2].detach()
                B, C, H, W = o.shape
                nc = C // 3 - 5
                o_view = o.view(B, 3, 5 + nc, H, W)
                obj_max = o_view[:, :, 4].sigmoid().max().item()
                cls_max = o_view[:, :, 5:].sigmoid().max().item()
                wh = (2 * o_view[:, :, 2:4].sigmoid()).pow(2).mean().item()
            print(f"{step:>5} | {loss.item():>8.4f} | {obj_max:>8.4f} | {cls_max:>8.4f} | {wh:>10.3f}")

    # Final check
    model.eval()
    with torch.no_grad():
        outs = model(imgs)
    print(f"\n=== FINAL EVAL on training batch ===")
    for s, o in enumerate(outs):
        B, C, H, W = o.shape
        nc = C // 3 - 5
        o_view = o.view(B, 3, 5 + nc, H, W)
        obj = o_view[:, :, 4].sigmoid()
        cls = o_view[:, :, 5:].sigmoid()
        score = obj.unsqueeze(2) * cls
        print(f"  Scale {s}: obj_max={obj.max():.3f}  score>0.5: {(score>0.5).sum()} cells")

    if args.save:
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "epoch": -1,
            "best_val_loss": float(loss.item()),
        }, args.save)
        print(f"\nSaved overfit checkpoint to {args.save}")

    print("\n→ Healthy: loss < 0.1, obj_max > 0.9, có vài cells với score > 0.5")
    print("→ Nếu loss kẹt ở > 1.0: có bug ở loss/model.")


if __name__ == "__main__":
    main()
