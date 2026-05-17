"""Backbone cho kiến trúc YOLO-style.

Backbone trích xuất đặc trưng từ ảnh đầu vào ở nhiều scale khác nhau.
Thiết kế đơn giản dựa trên CSP (Cross Stage Partial) blocks — lấy cảm hứng
từ DarkNet/CSPDarkNet nhưng rút gọn để dễ hiểu và mở rộng.

Output: ba feature map ở các scale khác nhau (stride 8, 16, 32)
để phục vụ cho multi-scale detection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34Backbone(nn.Module):
    """Backbone YOLO dùng ResNet-34 pretrained ImageNet.

    Trích feature map từ layer2/layer3/layer4 → P3/P4/P5 với stride 8/16/32.
    Số kênh output là (128, 256, 512) — khớp với neck mặc định, không cần
    sửa downstream.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        rn = resnet34(weights=weights)
        # Stem: conv1 + bn1 + relu + maxpool → stride 4
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1   # stride 4, 64 ch
        self.layer2 = rn.layer2   # stride 8, 128 ch  → P3
        self.layer3 = rn.layer3   # stride 16, 256 ch → P4
        self.layer4 = rn.layer4   # stride 32, 512 ch → P5

        self.out_channels = (128, 256, 512)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        return p3, p4, p5


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


class CSPBlock(nn.Module):
    """Cross Stage Partial block đơn giản.

    Chia feature map thành hai nhánh, một nhánh đi qua các bottleneck layers,
    sau đó ghép lại (concatenate) để giữ được gradient flow tốt hơn.

    Args:
        channels: Số kênh đầu vào và đầu ra.
        num_bottlenecks: Số lượng bottleneck layers trong nhánh chính.
    """

    def __init__(self, channels: int, num_bottlenecks: int = 1) -> None:
        super().__init__()
        mid = channels // 2
        self.split_conv1 = ConvBnSiLU(channels, mid, kernel=1)
        self.split_conv2 = ConvBnSiLU(channels, mid, kernel=1)

        self.bottlenecks = nn.Sequential(*[
            nn.Sequential(
                ConvBnSiLU(mid, mid, kernel=1),
                ConvBnSiLU(mid, mid, kernel=3),
            )
            for _ in range(num_bottlenecks)
        ])

        self.merge_conv = ConvBnSiLU(mid * 2, channels, kernel=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.bottlenecks(self.split_conv1(x))
        branch2 = self.split_conv2(x)
        return self.merge_conv(torch.cat([branch1, branch2], dim=1))


class YOLOBackbone(nn.Module):
    """Backbone kiểu DarkNet đơn giản cho YOLO.

    Trích xuất feature map ở ba scale (stride 8, 16, 32) để hỗ trợ
    detect vật thể nhỏ, trung bình, và lớn.

    Args:
        in_channels: Số kênh ảnh đầu vào (3 cho RGB).
        base_channels: Số kênh cơ sở, các tầng sau nhân đôi.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256
        c5 = base_channels * 16  # 512

        # Stem: giảm kích thước nhanh
        self.stem = ConvBnSiLU(in_channels, c1, kernel=3, stride=2)

        # Stage 1 → stride 4
        self.stage1 = nn.Sequential(
            ConvBnSiLU(c1, c2, kernel=3, stride=2),
            CSPBlock(c2, num_bottlenecks=1),
        )

        # Stage 2 → stride 8 (output scale nhỏ — detect vật thể lớn, gần)
        self.stage2 = nn.Sequential(
            ConvBnSiLU(c2, c3, kernel=3, stride=2),
            CSPBlock(c3, num_bottlenecks=2),
        )

        # Stage 3 → stride 16 (output scale trung bình)
        self.stage3 = nn.Sequential(
            ConvBnSiLU(c3, c4, kernel=3, stride=2),
            CSPBlock(c4, num_bottlenecks=2),
        )

        # Stage 4 → stride 32 (output scale lớn — detect vật thể nhỏ, xa)
        self.stage4 = nn.Sequential(
            ConvBnSiLU(c4, c5, kernel=3, stride=2),
            CSPBlock(c5, num_bottlenecks=1),
        )

        # Lưu lại số kênh output ở mỗi scale để Neck biết kết nối
        self.out_channels = (c3, c4, c5)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Trích xuất feature map ở 3 scale.

        Args:
            x: Ảnh đầu vào (batch, 3, H, W).

        Returns:
            Tuple gồm 3 feature map: (P3, P4, P5) tương ứng stride (8, 16, 32).
        """
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # stride 8
        p4 = self.stage3(p3)  # stride 16
        p5 = self.stage4(p4)  # stride 32
        return p3, p4, p5
