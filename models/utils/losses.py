"""Các hàm loss dùng cho object detection.

Module này chứa:
- FasterRCNNLoss: Placeholder — Faster R-CNN của torchvision tự tính loss bên trong,
  nhưng nếu cần custom thì có thể mở rộng tại đây.
- YOLOLoss: Loss cho kiến trúc YOLO-style (objectness + classification + bbox regression).
- SetCriterion: Loss theo kiểu set prediction của DETR, sử dụng Hungarian matching.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .box_ops import box_iou, generalized_box_iou, cxcywh_to_xyxy


# ---------------------------------------------------------------------------
# Faster R-CNN Loss
# ---------------------------------------------------------------------------

class FasterRCNNLoss(nn.Module):
    """Placeholder cho loss của Faster R-CNN.

    Trong torchvision, Faster R-CNN đã tự tính loss (rpn_loss + roi_loss)
    khi gọi model(images, targets) ở chế độ training.
    Class này chỉ để dự phòng nếu sau này cần custom loss riêng.
    """

    def forward(self, model_output: dict, targets: list[dict]) -> dict[str, Tensor]:
        # TODO: Triển khai custom loss nếu cần override loss mặc định của torchvision.
        #   Ví dụ: thay đổi trọng số giữa rpn_loss và roi_loss,
        #   hoặc thêm loss phụ (auxiliary loss).
        raise NotImplementedError(
            "Faster R-CNN dùng loss tích hợp sẵn trong torchvision. "
            "Override ở đây nếu cần custom."
        )


# ---------------------------------------------------------------------------
# YOLO Loss
# ---------------------------------------------------------------------------

class YOLOLoss(nn.Module):
    """Loss cho kiến trúc YOLO-style.

    Bao gồm ba thành phần chính:
    1. Objectness loss: BCE — dự đoán ô nào chứa vật thể.
    2. Classification loss: BCE hoặc CE — phân loại lớp vật thể.
    3. Bounding box regression loss: CIoU/GIoU — hồi quy toạ độ box.

    Args:
        num_classes: Số lớp vật thể (không tính background).
        lambda_obj: Trọng số cho objectness loss.
        lambda_cls: Trọng số cho classification loss.
        lambda_box: Trọng số cho bounding box loss.
    """

    def __init__(
        self,
        num_classes: int,
        lambda_obj: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_box: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box

        self.bce_obj = nn.BCEWithLogitsLoss()
        self.bce_cls = nn.BCEWithLogitsLoss()

    def forward(self, predictions: Tensor, targets: Tensor) -> dict[str, Tensor]:
        """Tính loss cho một batch.

        Args:
            predictions: Output từ detection head.
            targets: Ground-truth labels đã được encode phù hợp.

        Returns:
            Dict chứa từng thành phần loss và tổng loss.
        """
        # TODO: Triển khai chi tiết khi detection head và anchor/decode đã hoàn chỉnh.
        #   1. Decode predictions thành (tx, ty, tw, th, objectness, class_logits).
        #   2. Gán target cho từng anchor/grid cell (positive/negative matching).
        #   3. Tính objectness loss, classification loss, bbox regression loss.
        #   4. Trả về dict {"obj": ..., "cls": ..., "box": ..., "total": ...}.
        raise NotImplementedError(
            "YOLOLoss cần detection head hoàn chỉnh để triển khai. Xem TODO bên trên."
        )


# ---------------------------------------------------------------------------
# DETR SetCriterion
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    """Loss theo kiểu set prediction cho DETR.

    Sử dụng Hungarian matching để ghép cặp 1-1 giữa predictions và ground truth,
    sau đó tính loss trên các cặp đã ghép.

    Các thành phần loss:
    1. Classification loss: Cross-entropy (có trọng số cho lớp "no object").
    2. Bounding box L1 loss: Sai số tuyệt đối trên toạ độ box.
    3. GIoU loss: Generalized IoU giữa predicted box và target box.

    Args:
        num_classes: Số lớp vật thể (không tính "no object").
        matcher: Module thực hiện Hungarian matching.
        weight_ce: Trọng số cho classification loss.
        weight_bbox: Trọng số cho L1 bbox loss.
        weight_giou: Trọng số cho GIoU loss.
        eos_coef: Trọng số giảm cho lớp "no object" trong cross-entropy
                  (vì đa số queries không khớp với object nào).
    """

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_ce: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        eos_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_ce = weight_ce
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Tính tổng loss cho một batch.

        Args:
            outputs: Dict chứa:
                - "pred_logits": (batch, num_queries, num_classes + 1)
                - "pred_boxes":  (batch, num_queries, 4) ở định dạng cxcywh chuẩn hoá.
            targets: Danh sách dict cho mỗi ảnh, mỗi dict chứa:
                - "labels": (num_objects,)
                - "boxes":  (num_objects, 4) ở định dạng cxcywh chuẩn hoá.

        Returns:
            Dict chứa từng thành phần loss và tổng loss.
        """
        # TODO: Triển khai chi tiết khi matcher (Hungarian) đã hoàn chỉnh.
        #   1. Gọi self.matcher(outputs, targets) để lấy danh sách cặp (pred_idx, tgt_idx).
        #   2. Tính classification loss bằng cross-entropy có trọng số empty_weight.
        #   3. Tính L1 loss trên bbox.
        #   4. Tính GIoU loss.
        #   5. Trả về dict {"ce": ..., "bbox": ..., "giou": ..., "total": ...}.
        raise NotImplementedError(
            "SetCriterion cần Hungarian matcher hoàn chỉnh. Xem TODO bên trên."
        )
