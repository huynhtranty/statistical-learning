"""Backbone cho Faster R-CNN.

Sử dụng ResNet-50 + FPN từ torchvision làm backbone mặc định.
Có thể mở rộng để hỗ trợ các backbone khác (ResNet-101, MobileNet, v.v.).
"""
from __future__ import annotations

import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def build_backbone(
    backbone_name: str = "resnet50",
    pretrained: bool = True,
    trainable_layers: int = 3,
):
    """Tạo backbone có Feature Pyramid Network (FPN) cho Faster R-CNN.

    Args:
        backbone_name: Tên backbone (ví dụ: "resnet50", "resnet101").
        pretrained: Dùng trọng số đã pretrain trên ImageNet hay không.
        trainable_layers: Số lớp cuối của backbone được phép train.
            Giá trị từ 0 (đóng băng toàn bộ) đến 5 (train toàn bộ).

    Returns:
        Backbone kèm FPN, output là dict các feature map ở nhiều scale.
    """
    weights = "DEFAULT" if pretrained else None
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=weights,
        trainable_layers=trainable_layers,
    )
    # TODO: Hỗ trợ thêm các backbone khác ngoài ResNet (ví dụ: MobileNetV3, EfficientNet).
    # TODO: Cho phép cấu hình FPN chi tiết hơn (số kênh, extra blocks, v.v.).
    return backbone
