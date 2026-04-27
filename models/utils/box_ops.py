"""Các phép toán trên bounding box dùng chung cho tất cả model.

Hỗ trợ hai định dạng phổ biến:
- xyxy: (x_min, y_min, x_max, y_max)
- cxcywh: (center_x, center_y, width, height)
"""
from __future__ import annotations

import torch
from torch import Tensor


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Tính IoU (Intersection over Union) giữa hai tập bounding box.

    Cả hai đầu vào đều ở định dạng xyxy (x_min, y_min, x_max, y_max).

    Args:
        boxes1: Tensor kích thước (N, 4).
        boxes2: Tensor kích thước (M, 4).

    Returns:
        Tensor kích thước (N, M) chứa giá trị IoU cho mỗi cặp box.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Tính Generalized IoU (GIoU) giữa hai tập bounding box.

    GIoU bổ sung thêm phần phạt dựa trên bounding box bao ngoài (enclosing box),
    giúp gradient ổn định hơn khi hai box không giao nhau.

    Cả hai đầu vào đều ở định dạng xyxy.

    Args:
        boxes1: Tensor kích thước (N, 4).
        boxes2: Tensor kích thước (M, 4).

    Returns:
        Tensor kích thước (N, M) chứa giá trị GIoU trong khoảng [-1, 1].
    """
    iou = box_iou(boxes1, boxes2)

    enclose_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])

    enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * (enclose_y2 - enclose_y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area1[:, None] + area2 - inter_area

    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    return giou


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Chuyển đổi bounding box từ định dạng (cx, cy, w, h) sang (x_min, y_min, x_max, y_max).

    Args:
        boxes: Tensor kích thước (..., 4) ở định dạng (center_x, center_y, width, height).

    Returns:
        Tensor cùng kích thước ở định dạng (x_min, y_min, x_max, y_max).
    """
    cx, cy, w, h = boxes.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Chuyển đổi bounding box từ định dạng (x_min, y_min, x_max, y_max) sang (cx, cy, w, h).

    Args:
        boxes: Tensor kích thước (..., 4) ở định dạng (x_min, y_min, x_max, y_max).

    Returns:
        Tensor cùng kích thước ở định dạng (center_x, center_y, width, height).
    """
    x_min, y_min, x_max, y_max = boxes.unbind(-1)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return torch.stack([cx, cy, w, h], dim=-1)
