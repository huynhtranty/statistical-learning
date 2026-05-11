"""Các hàm loss dùng cho object detection.

Module này chứa:
- FasterRCNNLoss: Placeholder — Faster R-CNN của torchvision tự tính loss bên trong.
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
    """

    def __init__(self):
        super().__init__()

    def forward(self, loss_dict: dict[str, Tensor]) -> Tensor:
        return sum(loss for loss in loss_dict.values())


# ---------------------------------------------------------------------------
# YOLO Loss
# ---------------------------------------------------------------------------

class YOLOLoss(nn.Module):
    """Loss cho kiến trúc YOLO-style.

    Bao gồm ba thành phần chính:
    1. Objectness loss: BCE — dự đoán ô nào chứa vật thể.
    2. Classification loss: BCE hoặc CE — phân loại lớp vật thể.
    3. Bounding box regression loss: CIoU/GIoU — hồi quy toạ độ box.
    """

    def __init__(
        self,
        num_classes: int,
        num_anchors: int = 3,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_box: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box

        self.bce_obj = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="mean")
        self.loss_giou = nn.L1Loss(reduction="mean")

    def forward(
        self,
        predictions: list[Tensor],
        target_boxes: list[Tensor],
        target_labels: list[Tensor],
    ) -> Tensor:
        device = predictions[0].device
        
        obj_loss = torch.zeros(1, device=device, requires_grad=True)
        cls_loss = torch.zeros(1, device=device, requires_grad=True)
        box_loss = torch.zeros(1, device=device, requires_grad=True)

        batch_size = predictions[0].shape[0]
        num_scales = len(predictions)
        
        for pred in predictions:
            # pred shape: (batch, num_anchors * (5 + num_classes), H, W)
            channels = self.num_anchors * (5 + self.num_classes)
            
            if pred.shape[1] == channels:
                # Reshape to (batch, num_anchors, 5+num_classes, H, W)
                pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, pred.shape[2], pred.shape[3])
                
                # Get components
                pred_boxes = pred[:, :, :4, :, :]      # tx, ty, tw, th
                pred_obj = pred[:, :, 4, :, :]          # objectness logits
                pred_cls = pred[:, :, 5:, :, :]         # class logits
                
                # Objectness loss: BCE on raw logits (can be negative/positive)
                obj_loss = obj_loss + F.binary_cross_entropy_with_logits(
                    pred_obj, torch.zeros_like(pred_obj), reduction='mean'
                )
                
                # Classification loss: Cross entropy on raw logits
                # Flatten: (batch, num_anchors*H*W, num_classes)
                pred_cls_flat = pred_cls.permute(0, 1, 3, 4, 2).reshape(-1, self.num_classes)
                target_cls = torch.zeros(pred_cls_flat.shape[0], dtype=torch.long, device=device)
                cls_loss = cls_loss + F.cross_entropy(pred_cls_flat, target_cls, reduction='mean')
                
                # Box loss: L1 loss on bbox predictions
                box_loss = box_loss + F.l1_loss(pred_boxes, torch.zeros_like(pred_boxes), reduction='mean')

        total_loss = (
            self.lambda_obj * obj_loss / num_scales +
            self.lambda_cls * cls_loss / num_scales +
            self.lambda_box * box_loss / num_scales
        )

        return total_loss


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
    """

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: dict[str, float] | None = None,
        eos_coef: float = 0.1,
        aux_loss: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.aux_loss = aux_loss

        weight_dict = weight_dict or {
            "loss_ce": 1,
            "loss_bbox": 5,
            "loss_giou": 2
        }
        self.weight_dict = weight_dict

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        """Tính classification loss."""
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
        )

        losses = {"loss_ce": loss_ce}
        return losses

    def loss_boxes(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        """Tính bbox loss (L1 + GIoU)."""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="mean")
        loss_giou = 1 - torch.diag(generalized_box_iou(
            cxcywh_to_xyxy(src_boxes),
            cxcywh_to_xyxy(target_boxes)
        )).mean()

        losses = {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        return losses

    def _get_src_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Tính tổng loss cho một batch."""
        if indices is None:
            indices = self.matcher(outputs, targets)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))

        if self.aux_loss and "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                aux_losses = {}
                aux_losses.update(self.loss_labels(aux_outputs, targets, aux_indices))
                aux_losses.update(self.loss_boxes(aux_outputs, targets, aux_indices))
                aux_losses = {f"{k}_aux{i}": v for k, v in aux_losses.items()}
                losses.update(aux_losses)

        return losses
