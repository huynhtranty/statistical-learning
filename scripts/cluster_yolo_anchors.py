"""Cluster YOLO anchors từ COCO annotations theo IoU distance.

Ví dụ:
    python scripts/cluster_yolo_anchors.py \
        --ann_file data/annotations/train.json \
        --img_size 640 \
        --num_scales 3 \
        --anchors_per_scale 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster YOLO anchors từ COCO train annotations.")
    p.add_argument("--ann_file", type=Path, required=True, help="Path tới COCO train.json")
    p.add_argument("--img_size", type=int, default=640, help="Input image size để normalize anchors")
    p.add_argument("--num_scales", type=int, default=3, help="Số scale heads (thường 3)")
    p.add_argument("--anchors_per_scale", type=int, default=3, help="Số anchors mỗi scale")
    p.add_argument("--iters", type=int, default=200, help="Số vòng lặp kmeans")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def wh_iou(box_wh: np.ndarray, cluster_wh: np.ndarray) -> np.ndarray:
    """IoU giữa box_wh (N,2) và cluster_wh (K,2) tại cùng center."""
    inter_w = np.minimum(box_wh[:, None, 0], cluster_wh[None, :, 0])
    inter_h = np.minimum(box_wh[:, None, 1], cluster_wh[None, :, 1])
    inter = inter_w * inter_h
    box_area = box_wh[:, 0:1] * box_wh[:, 1:2]
    cluster_area = cluster_wh[:, 0] * cluster_wh[:, 1]
    union = box_area + cluster_area[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def kmeans_anchors(
    box_wh: np.ndarray,
    k: int,
    iters: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Kmeans theo khoảng cách 1 - IoU."""
    if box_wh.shape[0] < k:
        raise ValueError(f"Số box ({box_wh.shape[0]}) nhỏ hơn số clusters ({k}).")

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(box_wh.shape[0], size=k, replace=False)
    centers = box_wh[init_idx].copy()

    for _ in range(iters):
        ious = wh_iou(box_wh, centers)
        dist = 1.0 - ious
        assignments = dist.argmin(axis=1)

        new_centers = centers.copy()
        for c in range(k):
            points = box_wh[assignments == c]
            if points.shape[0] > 0:
                new_centers[c] = points.mean(axis=0)

        delta = np.abs(new_centers - centers).mean()
        centers = new_centers
        if delta < 1e-6:
            break

    return centers


def main() -> None:
    args = parse_args()

    with args.ann_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    img_meta = {int(img["id"]): (float(img["width"]), float(img["height"])) for img in coco["images"]}

    box_wh = []
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        img_id = int(ann["image_id"])
        img_w, img_h = img_meta.get(img_id, (float(args.img_size), float(args.img_size)))
        w_norm = float(w) / max(img_w, 1.0)
        h_norm = float(h) / max(img_h, 1.0)
        if w_norm <= 0 or h_norm <= 0:
            continue
        box_wh.append([w_norm, h_norm])

    if not box_wh:
        raise RuntimeError("Không có bbox hợp lệ để cluster anchors.")

    box_wh_np = np.asarray(box_wh, dtype=np.float32)
    total_anchors = int(args.num_scales) * int(args.anchors_per_scale)
    centers = kmeans_anchors(box_wh_np, total_anchors, iters=args.iters, seed=args.seed)

    # Sort theo area tăng dần rồi chia cho các scale (small -> large).
    areas = centers[:, 0] * centers[:, 1]
    order = np.argsort(areas)
    centers = centers[order]

    grouped = []
    ptr = 0
    for _ in range(args.num_scales):
        scale_anchors = centers[ptr:ptr + args.anchors_per_scale]
        ptr += args.anchors_per_scale
        grouped.append(scale_anchors.tolist())

    # Mean best IoU để tham khảo quality anchor fit.
    ious = wh_iou(box_wh_np, centers)
    best_iou = ious.max(axis=1)
    mean_best_iou = float(best_iou.mean())

    print("[Anchors] Normalized anchors (w,h):")
    for i, anchors in enumerate(grouped):
        row = ", ".join([f"[{w:.4f}, {h:.4f}]" for w, h in anchors])
        print(f"  P{i+3}: [{row}]")
    print(f"[Anchors] Mean best IoU: {mean_best_iou:.4f}")
    print("\n[YAML snippet]")
    print("anchors:")
    for anchors in grouped:
        row = ", ".join([f"[{w:.4f}, {h:.4f}]" for w, h in anchors])
        print(f"  - [{row}]")


if __name__ == "__main__":
    main()
