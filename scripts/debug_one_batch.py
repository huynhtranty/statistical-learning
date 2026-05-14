#!/usr/bin/env python3
"""In thống kê 1 batch: image stats, bbox stats, model output stats, loss components.

Usage:
    python scripts/debug_one_batch.py --data_root data --weights weights/yolo.pt
    python scripts/debug_one_batch.py --data_root data   # untrained model
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
    p.add_argument("--weights", default=None, help="optional checkpoint to load")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    classes = get_class_names(Path(args.data_root) / "annotations")
    num_classes = len(classes)
    print(f"[INFO] {num_classes} classes: {classes}")

    ds = CocoDetection(
        img_folder=Path(args.data_root) / "images" / "train",
        ann_file=Path(args.data_root) / "annotations" / "train.json",
        classes=classes,
        img_size=args.img_size,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # ---- Lấy 1 batch ----
    imgs, targets = next(iter(dl))
    imgs = imgs.to(device)
    print(f"\n=== BATCH STATS ===")
    print(f"images.shape = {tuple(imgs.shape)}   dtype={imgs.dtype}")
    print(f"images range: min={imgs.min():.4f}, max={imgs.max():.4f}, mean={imgs.mean():.4f}")
    print(f"  (kỳ vọng [0, 1] cho YOLO/Faster R-CNN)")

    for i, t in enumerate(targets):
        b = t["boxes"]
        l = t["labels"]
        if b.numel() == 0:
            print(f"  img[{i}]: 0 objects")
            continue
        ws = b[:, 2] - b[:, 0]
        hs = b[:, 3] - b[:, 1]
        print(f"  img[{i}]: {b.shape[0]} obj, "
              f"x range [{b[:,0].min():.0f}, {b[:,2].max():.0f}], "
              f"y range [{b[:,1].min():.0f}, {b[:,3].max():.0f}], "
              f"w mean={ws.mean():.0f}, h mean={hs.mean():.0f}, "
              f"labels={l.tolist()}")
        if (ws <= 0).any() or (hs <= 0).any():
            print(f"     ⚠️ DEGENERATE BOX: w/h ≤ 0")
        if l.min() < 0 or l.max() >= num_classes:
            print(f"     ⚠️ INVALID LABEL: range [{l.min()}, {l.max()}] vs num_classes={num_classes}")

    # ---- Build model + load checkpoint ----
    print(f"\n=== MODEL ===")
    model = build_yolo(num_classes=num_classes, pretrained_backbone=(args.weights is None))
    if args.weights:
        ck = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"Loaded checkpoint epoch={ck.get('epoch')}, val_loss={ck.get('best_val_loss')}")
    model.to(device).eval()

    with torch.no_grad():
        outs = model(imgs)

    print(f"\n=== MODEL OUTPUT (eval mode) ===")
    for s, o in enumerate(outs):
        B, C, H, W = o.shape
        nc = C // 3 - 5
        o_view = o.view(B, 3, 5 + nc, H, W)
        obj_sig = o_view[:, :, 4].sigmoid()
        cls_sig = o_view[:, :, 5:].sigmoid()
        # YOLOv5 wh
        wh = (2 * o_view[:, :, 2:4].sigmoid()).pow(2)
        print(f"  Scale {s} ({H}x{W}):  obj range [{obj_sig.min():.4f}, {obj_sig.max():.4f}],  "
              f"cls max={cls_sig.max():.4f},  wh decoded mean=[{wh[:,:,0].mean():.3f}, {wh[:,:,1].mean():.3f}]")

    # ---- Loss components ----
    print(f"\n=== LOSS ===")
    model.train()
    crit = YOLOLoss(num_classes=num_classes, num_anchors=3, image_size=args.img_size)
    outs = model(imgs)
    target_boxes = [t["boxes"].to(device) for t in targets]
    target_labels = [t["labels"].to(device) for t in targets]
    loss = crit(outs, target_boxes, target_labels)
    print(f"Total loss: {loss.item():.4f}")

    loss.backward()
    # Check gradient flow on box head
    for name, p in model.head.heads[2].pred.named_parameters():
        if "bias" in name:
            g = p.grad.view(3, 5 + num_classes)
            print(f"Head P5 bias gradient:")
            print(f"  tx: {g[:,0].abs().mean():.4f}  ty: {g[:,1].abs().mean():.4f}")
            print(f"  tw: {g[:,2].abs().mean():.4f}  th: {g[:,3].abs().mean():.4f}  (cao = đang học wh)")
            print(f"  obj: {g[:,4].abs().mean():.4f}  cls: {g[:,5:].abs().mean():.4f}")

    print("\n→ Healthy signs:")
    print("   - images in [0,1]")
    print("   - labels in [0, num_classes-1]")
    print("   - bbox không degenerate")
    print("   - obj range chứa giá trị > 0.05 (model có firing)")
    print("   - tw/th gradient > 0.001 (đang học box size)")


if __name__ == "__main__":
    main()
