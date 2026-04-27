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
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 640,
    max_size: int = 640,
) -> FasterRCNN:
    """Tạo mô hình Faster R-CNN với số lớp tuỳ chỉnh.

    Args:
        num_classes: Số lớp vật thể (bao gồm cả background).
            Ví dụ: 5 lớp vật thể + 1 background = 6.
        pretrained_backbone: Dùng backbone đã pretrain trên COCO hay không.
        trainable_backbone_layers: Số lớp cuối của backbone được phép train (0–5).
        min_size: Kích thước nhỏ nhất của ảnh đầu vào sau khi resize.
        max_size: Kích thước lớn nhất của ảnh đầu vào sau khi resize.

    Returns:
        Mô hình FasterRCNN sẵn sàng để train hoặc inference.
    """
    weights = "DEFAULT" if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )

    # Thay thế box predictor head để phù hợp số lớp của dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # TODO: Thay đổi anchor sizes/aspect ratios nếu dataset có vật thể kích thước đặc biệt.
    # TODO: Thêm mask head nếu muốn mở rộng sang instance segmentation (Mask R-CNN).
    # TODO: Tuỳ chỉnh RPN (số anchor mỗi vị trí, ngưỡng NMS, v.v.).

    return model
