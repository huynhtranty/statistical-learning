"""Kiểm tra kiến trúc Faster R-CNN và skeleton cho training.

Script này chủ yếu dùng để:
- Kiểm tra model build thành công hay không.
- Forward thử với dummy input.
- In ra tóm tắt kiến trúc.

Phần training loop thật sẽ được triển khai sau.

Cách chạy:
    python models/faster_rcnn/train.py --num_classes 6
    python models/faster_rcnn/train.py --num_classes 6 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Thêm thư mục gốc của project vào sys.path để import được models.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.faster_rcnn.model import build_faster_rcnn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kiểm tra kiến trúc Faster R-CNN.")
    p.add_argument("--num_classes", type=int, default=6,
                    help="Số lớp (bao gồm background). Mặc định: 6.")
    p.add_argument("--device", type=str, default="cpu",
                    help="Thiết bị chạy model (cpu hoặc cuda). Mặc định: cpu.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"[Faster R-CNN] Đang build model với num_classes={args.num_classes}...")
    model = build_faster_rcnn(num_classes=args.num_classes)
    model.to(device)

    # Đếm tổng số tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Faster R-CNN] Tổng tham số:       {total_params:,}")
    print(f"[Faster R-CNN] Tham số trainable:   {trainable_params:,}")

    # Forward thử với dummy input ở chế độ eval
    model.eval()
    dummy_images = [torch.randn(3, 640, 640, device=device)]
    with torch.no_grad():
        outputs = model(dummy_images)
    print(f"[Faster R-CNN] Forward thành công! Số detection đầu ra: {len(outputs[0]['boxes'])}")
    print(f"[Faster R-CNN] Keys trong output: {list(outputs[0].keys())}")

    # TODO: Triển khai DataLoader với COCO annotations.
    # TODO: Triển khai training loop (optimizer, scheduler, epoch loop).
    # TODO: Triển khai validation sau mỗi epoch.
    # TODO: Lưu checkpoint tốt nhất.
    print("\n[Faster R-CNN] Model build thành công. Training loop chưa được triển khai (xem TODO).")


if __name__ == "__main__":
    main()
