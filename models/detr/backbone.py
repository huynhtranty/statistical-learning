"""Backbone cho DETR.

Sử dụng CNN (ResNet-50 mặc định) để trích xuất feature map,
sau đó giảm số kênh bằng một lớp conv 1×1 để phù hợp với
hidden dimension của transformer.

Output: feature map (batch, hidden_dim, H/32, W/32) kèm position encoding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights


class DETRBackbone(nn.Module):
    """CNN backbone cho DETR.

    Trích xuất feature map từ layer cuối cùng của ResNet-50 (stride 32),
    rồi chiếu xuống hidden_dim kênh bằng conv 1×1.

    Args:
        hidden_dim: Số chiều ẩn của transformer (mặc định 256).
        pretrained: Dùng trọng số pretrained trên ImageNet hay không.
    """

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Lấy tất cả các layer trừ avgpool và fc cuối cùng
        self.body = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Chiếu từ 2048 kênh (output ResNet-50 layer4) xuống hidden_dim
        self.proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # TODO: Hỗ trợ backbone khác (ResNet-101, ResNeXt, v.v.).
        # TODO: Cho phép đóng băng một số layer đầu để tiết kiệm bộ nhớ khi fine-tune.

    def forward(self, x: Tensor) -> Tensor:
        """Trích xuất feature map từ ảnh đầu vào.

        Args:
            x: Ảnh đầu vào (batch, 3, H, W).

        Returns:
            Feature map (batch, hidden_dim, H/32, W/32).
        """
        features = self.body(x)
        features = self.proj(features)
        return features
