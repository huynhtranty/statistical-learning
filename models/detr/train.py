"""Kiểm tra kiến trúc DETR và skeleton cho training.

Script này chủ yếu dùng để:
- Kiểm tra model build thành công hay không.
- Forward thử với dummy input.
- In ra tóm tắt kiến trúc và kích thước output.

Phần training loop thật sẽ được triển khai sau.

Cách chạy:
    python models/detr/train.py --num_classes 5
    python models/detr/train.py --num_classes 5 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.detr.model import build_detr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kiểm tra kiến trúc DETR.")
    p.add_argument("--num_classes", type=int, default=5,
                    help="Số lớp vật thể (không tính 'no object'). Mặc định: 5.")
    p.add_argument("--num_queries", type=int, default=100,
                    help="Số object queries. Mặc định: 100.")
    p.add_argument("--device", type=str, default="cpu",
                    help="Thiết bị chạy model (cpu hoặc cuda). Mặc định: cpu.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"[DETR] Đang build model với num_classes={args.num_classes}, "
          f"num_queries={args.num_queries}...")
    model = build_detr(num_classes=args.num_classes, num_queries=args.num_queries)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DETR] Tổng tham số:       {total_params:,}")
    print(f"[DETR] Tham số trainable:   {trainable_params:,}")

    # Forward thử với dummy input
    dummy = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        outputs = model(dummy)

    print(f"[DETR] Forward thành công!")
    print(f"  pred_logits shape: {tuple(outputs['pred_logits'].shape)}")
    print(f"  pred_boxes shape:  {tuple(outputs['pred_boxes'].shape)}")

    # TODO: Triển khai DataLoader với COCO annotations.
    # TODO: Triển khai training loop (AdamW optimizer, scheduler, gradient clipping).
    # TODO: Tích hợp HungarianMatcher + SetCriterion cho loss.
    # TODO: Triển khai validation sau mỗi epoch.
    # TODO: Lưu checkpoint tốt nhất.
    # TODO: Hỗ trợ auxiliary loss (dự đoán ở mỗi decoder layer).
    print("\n[DETR] Model build thành công. Training loop chưa được triển khai (xem TODO).")


if __name__ == "__main__":
    main()
