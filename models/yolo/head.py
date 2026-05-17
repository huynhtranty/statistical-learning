"""Detection Head cho kiến trúc YOLO-style.

Head nhận feature map từ Neck và dự đoán:
- Bounding box (tx, ty, tw, th) cho mỗi anchor tại mỗi vị trí grid.
- Objectness score: xác suất ô đó chứa vật thể.
- Class logits: xác suất thuộc từng lớp vật thể.

Output ở dạng (batch, num_anchors_per_cell * (5 + num_classes), H, W)
cho mỗi feature map scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ConvBnSiLU(nn.Module):
    """Khối cơ bản: Conv2d + BatchNorm + SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class SingleScaleHead(nn.Module):
    """Head cho một scale cụ thể.

    Gồm vài lớp conv để tinh chỉnh feature, sau đó
    một lớp conv 1×1 cuối để ra prediction.

    Args:
        in_ch: Số kênh đầu vào (từ Neck).
        num_anchors: Số anchor box mỗi vị trí grid.
        num_classes: Số lớp vật thể.
    """

    def __init__(self, in_ch: int, num_anchors: int, num_classes: int) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Mỗi anchor dự đoán: 4 (bbox) + 1 (objectness) + num_classes
        out_ch = num_anchors * (5 + num_classes)

        self.conv = nn.Sequential(
            ConvBnSiLU(in_ch, in_ch, kernel=3),
            ConvBnSiLU(in_ch, in_ch, kernel=3),
        )
        self.pred = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        # Bias prior init theo YOLOv5 parameterization mới:
        #   pred_xy = 2*sigmoid(t) - 0.5     ∈ [-0.5, 1.5]   (offset cell)
        #   pred_wh = (2*sigmoid(t))**2      ∈ [0, 4]        (fraction of image)
        # → muốn pred_wh ≈ 0.22 (mean object size) ở init:
        #   (2*sigmoid(b))^2 = 0.22 → sigmoid(b) ≈ 0.235 → b ≈ -1.18
        # → muốn pred_xy ≈ 0.5 (center cell): 2*sigmoid(0) - 0.5 = 0.5 ✓ (bias=0).
        # - obj: sigmoid(-4.6) ≈ 0.01 (negative prior, tránh collapse).
        # - cls: sigmoid(b) = 1/num_classes (uniform prior).
        import math
        stride_per_anchor = 5 + num_classes
        with torch.no_grad():
            bias = self.pred.bias.view(num_anchors, stride_per_anchor)
            bias.zero_()
            bias[:, 2] = -1.18  # tw → (2*sigmoid(-1.18))^2 ≈ 0.22
            bias[:, 3] = -1.18  # th → ≈ 0.22
            bias[:, 4] = -4.6   # obj → sigmoid ≈ 0.01
            cls_prior = max(1.0 / max(num_classes, 1), 1e-3)
            bias[:, 5:] = -math.log((1.0 - cls_prior) / cls_prior)

    def forward(self, x: Tensor) -> Tensor:
        """Dự đoán detection cho một feature map.

        Args:
            x: Feature map (batch, in_ch, H, W).

        Returns:
            Tensor (batch, num_anchors * (5 + num_classes), H, W).
        """
        return self.pred(self.conv(x))


class YOLODetectionHead(nn.Module):
    """Multi-scale detection head cho YOLO.

    Nhận 3 feature map từ Neck và dự đoán bounding box + class
    ở từng scale.

    Args:
        in_channels: Tuple chứa số kênh của 3 feature map từ Neck.
        num_anchors: Số anchor box mỗi vị trí grid (mặc định 3).
        num_classes: Số lớp vật thể.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        num_anchors: int = 3,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.heads = nn.ModuleList([
            SingleScaleHead(ch, num_anchors, num_classes)
            for ch in in_channels
        ])

        # TODO: Định nghĩa anchor sizes cụ thể cho từng scale.
        #   Ví dụ: scale nhỏ (stride 8) → anchor nhỏ, scale lớn (stride 32) → anchor lớn.
        # TODO: Triển khai hàm decode_predictions() để chuyển raw output
        #   thành bounding box toạ độ thật (xmin, ymin, xmax, ymax).
        # TODO: Triển khai NMS (Non-Maximum Suppression) cho inference.

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> list[Tensor]:
        """Dự đoán detection ở tất cả scale.

        Args:
            features: Tuple (P3, P4, P5) từ Neck.

        Returns:
            Danh sách 3 tensor prediction, mỗi tensor có shape
            (batch, num_anchors * (5 + num_classes), H_i, W_i).
        """
        return [head(feat) for head, feat in zip(self.heads, features)]
