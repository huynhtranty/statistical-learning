"""Kiến trúc YOLO-style cho object detection.

YOLO (You Only Look Once) là mô hình one-stage detector:
- Backbone: trích xuất feature map ở nhiều scale.
- Neck: kết hợp feature map (FPN-style).
- Detection Head: dự đoán bounding box, objectness, và class logits.

Kiến trúc này là phiên bản đơn giản hoá, lấy cảm hứng từ YOLOv5/v8
nhưng rút gọn để dễ hiểu, dễ mở rộng, và phù hợp cho mục đích học tập.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .backbone import YOLOBackbone, ResNet34Backbone
from .neck import YOLONeck
from .head import YOLODetectionHead


class YOLO(nn.Module):
    """Mô hình YOLO hoàn chỉnh gồm backbone + neck + head.

    Args:
        num_classes: Số lớp vật thể (không tính background).
        in_channels: Số kênh ảnh đầu vào (3 cho RGB).
        base_channels: Số kênh cơ sở (chỉ dùng khi `use_csp_backbone=True`).
        num_anchors: Số anchor box mỗi vị trí grid.
        pretrained_backbone: Có download trọng số ImageNet cho ResNet-34 hay không.
            Khi load checkpoint của ta, đặt False (kiến trúc vẫn là ResNet-34,
            chỉ skip download weights vì sẽ load từ checkpoint).
        use_csp_backbone: Nếu True, dùng CSPDarkNet from-scratch thay vì ResNet-34.
            Chủ yếu phục vụ minh hoạ "custom backbone" trong báo cáo.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 3,
        base_channels: int = 32,
        num_anchors: int = 3,
        pretrained_backbone: bool = True,
        use_csp_backbone: bool = False,
    ) -> None:
        super().__init__()
        if use_csp_backbone:
            self.backbone = YOLOBackbone(in_channels=in_channels, base_channels=base_channels)
        else:
            self.backbone = ResNet34Backbone(pretrained=pretrained_backbone)
        self.neck = YOLONeck(in_channels=self.backbone.out_channels)
        self.head = YOLODetectionHead(
            in_channels=self.neck.out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass qua toàn bộ pipeline.

        Args:
            x: Ảnh đầu vào (batch, 3, H, W). H và W nên chia hết cho 32.

        Returns:
            Danh sách 3 tensor prediction ở 3 scale khác nhau.
            Mỗi tensor: (batch, num_anchors * (5 + num_classes), H_i, W_i).
        """
        features = self.backbone(x)
        features = self.neck(features)
        predictions = self.head(features)
        return predictions


def build_yolo(
    num_classes: int = 5,
    base_channels: int = 32,
    num_anchors: int = 3,
    pretrained_backbone: bool = True,
    use_csp_backbone: bool = False,
) -> YOLO:
    """Tạo mô hình YOLO.

    Args:
        num_classes: Số lớp vật thể (không tính background).
        base_channels: Số kênh cơ sở cho backbone CSP (chỉ dùng khi use_csp_backbone=True).
        num_anchors: Số anchor box mỗi vị trí grid.
        pretrained_backbone: True → ResNet-34 ImageNet pretrained.
            False → cùng kiến trúc ResNet-34 nhưng KHÔNG download weights
            (dùng khi sẽ load checkpoint của ta vào sau).
        use_csp_backbone: True → đổi sang CSPDarkNet from-scratch (custom backbone).

    Returns:
        Mô hình YOLO sẵn sàng để train hoặc inference.
    """
    return YOLO(
        num_classes=num_classes,
        base_channels=base_channels,
        num_anchors=num_anchors,
        pretrained_backbone=pretrained_backbone,
        use_csp_backbone=use_csp_backbone,
    )
