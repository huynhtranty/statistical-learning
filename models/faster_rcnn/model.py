"""Kiến trúc Faster R-CNN cho object detection.

Faster R-CNN là mô hình two-stage:
1. Region Proposal Network (RPN) đề xuất vùng có thể chứa vật thể.
2. ROI Head phân loại và tinh chỉnh bounding box cho từng vùng đề xuất.

Sử dụng torchvision.models.detection làm nền tảng, chỉ thay đổi
số lớp output (box predictor) để phù hợp với dataset riêng.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 640,
    max_size: int = 640,
) -> FasterRCNN:
    """Tạo mô hình Faster R-CNN, fine-tune từ checkpoint COCO-pretrained.

    Args:
        num_classes: Tổng số lớp bao gồm background (label 0).
            Ví dụ: 10 lớp vật thể + 1 background = 11.
        pretrained: Nếu True, load full detector COCO-pretrained
            (`fasterrcnn_resnet50_fpn(weights="DEFAULT")`) rồi thay thế
            box predictor cuối cùng cho num_classes của bài toán.
            Nếu False, chỉ ImageNet backbone (random head).
        trainable_backbone_layers: Số stage cuối của backbone được phép train (0–5).
        min_size, max_size: Range resize ảnh đầu vào.

    Returns:
        Mô hình FasterRCNN sẵn sàng fine-tune.
    """
    if pretrained:
        # Load full COCO detector — RPN, FPN, ROI heads đều đã train trên COCO.
        model = fasterrcnn_resnet50_fpn(
            weights="DEFAULT",
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )
        # Thay thế box predictor 91-class (COCO) bằng head num_classes của ta.
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        # Chỉ backbone ImageNet, head random — dùng khi reload checkpoint của ta.
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
        )

    return model
