"""Neck cho kiến trúc YOLO-style.

Neck kết hợp các feature map từ backbone ở nhiều scale khác nhau
để tạo ra feature map phong phú hơn cho detection head.

Thiết kế theo kiểu FPN (Feature Pyramid Network) đơn giản:
- Top-down pathway: upsample feature map lớn rồi cộng/ghép với feature map nhỏ hơn.
- Kết quả: 3 feature map đã được tăng cường ngữ cảnh ở mỗi scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class YOLONeck(nn.Module):
    """FPN-style neck cho YOLO.

    Nhận 3 feature map từ backbone (P3, P4, P5) và tạo ra
    3 feature map đã tăng cường để đưa vào detection head.

    Args:
        in_channels: Tuple chứa số kênh của (P3, P4, P5) từ backbone.
    """

    def __init__(self, in_channels: tuple[int, int, int]) -> None:
        super().__init__()
        c3, c4, c5 = in_channels

        # Nhánh top-down: P5 → P4
        self.reduce_p5 = ConvBnSiLU(c5, c4, kernel=1)
        self.fuse_p4 = ConvBnSiLU(c4 * 2, c4, kernel=3)

        # Nhánh top-down: P4 → P3
        self.reduce_p4 = ConvBnSiLU(c4, c3, kernel=1)
        self.fuse_p3 = ConvBnSiLU(c3 * 2, c3, kernel=3)

        # TODO: Thêm bottom-up pathway (PAN — Path Aggregation Network) để cải thiện
        #   khả năng truyền thông tin từ scale nhỏ lên scale lớn.
        # TODO: Thêm SPP (Spatial Pyramid Pooling) block ở đỉnh nếu cần.

        self.out_channels = (c3, c4, c5)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Kết hợp feature map từ nhiều scale.

        Args:
            features: Tuple (P3, P4, P5) từ backbone.

        Returns:
            Tuple (out_p3, out_p4, out_p5) — feature map đã tăng cường.
        """
        p3, p4, p5 = features

        # Top-down: P5 → P4
        p5_up = F.interpolate(self.reduce_p5(p5), size=p4.shape[2:], mode="nearest")
        p4_fused = self.fuse_p4(torch.cat([p4, p5_up], dim=1))

        # Top-down: P4 → P3
        p4_up = F.interpolate(self.reduce_p4(p4_fused), size=p3.shape[2:], mode="nearest")
        p3_fused = self.fuse_p3(torch.cat([p3, p4_up], dim=1))

        return p3_fused, p4_fused, p5

