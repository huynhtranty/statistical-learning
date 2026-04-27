"""Kiểm tra kiến trúc YOLO và skeleton cho training.

Script này chủ yếu dùng để:
- Kiểm tra model build thành công hay không.
- Forward thử với dummy input.
- In ra tóm tắt kiến trúc và kích thước output.

Phần training loop thật sẽ được triển khai sau.

Cách chạy:
    python models/yolo/train.py --num_classes 5
    python models/yolo/train.py --num_classes 5 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.yolo.model import build_yolo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kiểm tra kiến trúc YOLO.")
    p.add_argument("--num_classes", type=int, default=5,
                    help="Số lớp vật thể (không tính background). Mặc định: 5.")
    p.add_argument("--device", type=str, default="cpu",
                    help="Thiết bị chạy model (cpu hoặc cuda). Mặc định: cpu.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"[YOLO] Đang build model với num_classes={args.num_classes}...")
    model = build_yolo(num_classes=args.num_classes)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[YOLO] Tổng tham số:       {total_params:,}")
    print(f"[YOLO] Tham số trainable:   {trainable_params:,}")

    # Forward thử với dummy input
    dummy = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        outputs = model(dummy)

    print(f"[YOLO] Forward thành công! Số scale output: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: shape = {tuple(out.shape)}")

    # TODO: Triển khai DataLoader với annotations (YOLO format hoặc COCO).
    # TODO: Triển khai training loop (optimizer, scheduler, epoch loop).
    # TODO: Triển khai decode predictions và NMS cho inference.
    # TODO: Triển khai validation sau mỗi epoch.
    # TODO: Lưu checkpoint tốt nhất.
    print("\n[YOLO] Model build thành công. Training loop chưa được triển khai (xem TODO).")


if __name__ == "__main__":
    main()
