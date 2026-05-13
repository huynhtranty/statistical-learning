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
        image_size: int = 640,
        lambda_noobj: float = 0.1,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_box: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.image_size = float(image_size)
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box

        self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="none")
        self.box_loss_fn = nn.SmoothL1Loss(reduction="mean")

    def forward(
        self,
        predictions: list[Tensor],
        target_boxes: list[Tensor],
        target_labels: list[Tensor],
    ) -> Tensor:
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        num_scales = len(predictions)
        img_size = self.image_size

        obj_loss = torch.zeros((), device=device)
        cls_loss = torch.zeros((), device=device)
        box_loss = torch.zeros((), device=device)
        total_pos = 0

        for pred in predictions:
            _, channels, h, w = pred.shape
            expected = self.num_anchors * (5 + self.num_classes)
            if channels != expected:
                continue

            pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, h, w)
            pred_boxes = pred[:, :, :4, :, :]   # tx, ty, tw, th
            pred_obj = pred[:, :, 4, :, :]      # logits
            pred_cls = pred[:, :, 5:, :, :]     # logits

            obj_target = torch.zeros_like(pred_obj)
            pos_mask = torch.zeros_like(pred_obj, dtype=torch.bool)
            cls_target = torch.zeros_like(pred_cls)

            for b in range(batch_size):
                boxes = target_boxes[b]
                labels = target_labels[b]
                if boxes.numel() == 0:
                    continue

                boxes = boxes.to(device)
                labels = labels.to(device)

                cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5 / img_size).clamp(0.0, 1.0 - 1e-6)
                cy = ((boxes[:, 1] + boxes[:, 3]) * 0.5 / img_size).clamp(0.0, 1.0 - 1e-6)
                bw = ((boxes[:, 2] - boxes[:, 0]) / img_size).clamp(1e-6, 1.0)
                bh = ((boxes[:, 3] - boxes[:, 1]) / img_size).clamp(1e-6, 1.0)

                gx = cx * w
                gy = cy * h
                gi = torch.floor(gx).long().clamp(0, w - 1)
                gj = torch.floor(gy).long().clamp(0, h - 1)
                tx = gx - gi.float()
                ty = gy - gj.float()

                valid = (labels >= 0) & (labels < self.num_classes)
                if valid.sum() == 0:
                    continue

                gi = gi[valid]
                gj = gj[valid]
                tx = tx[valid]
                ty = ty[valid]
                bw = bw[valid]
                bh = bh[valid]
                labels = labels[valid]

                # Minimal assignment: mark all anchors at responsible cell positive.
                for a in range(self.num_anchors):
                    obj_target[b, a, gj, gi] = 1.0
                    pos_mask[b, a, gj, gi] = True
                    cls_target[b, a, labels, gj, gi] = 1.0

                    pred_xywh = pred_boxes[b, a, :, gj, gi].transpose(0, 1)
                    pred_xy = pred_xywh[:, :2].sigmoid()
                    pred_wh = pred_xywh[:, 2:].sigmoid()
                    tgt_xywh = torch.stack((tx, ty, bw, bh), dim=1)
                    box_loss = box_loss + self.box_loss_fn(
                        torch.cat((pred_xy, pred_wh), dim=1),
                        tgt_xywh,
                    )

                total_pos += int(gi.numel()) * self.num_anchors

            obj_elementwise = self.bce_obj(pred_obj, obj_target)
            pos_count = pos_mask.sum().item()
            neg_mask = ~pos_mask
            neg_count = neg_mask.sum().item()

            if pos_count > 0:
                obj_pos = obj_elementwise[pos_mask].mean()
            else:
                obj_pos = torch.zeros((), device=device)

            if neg_count > 0:
                obj_neg = obj_elementwise[neg_mask].mean()
            else:
                obj_neg = torch.zeros((), device=device)

            obj_loss = obj_loss + obj_pos + self.lambda_noobj * obj_neg

            # Class loss should be computed only on positive anchors/cells.
            # Broadcast pos mask to class dimension: (B, A, C, H, W)
            cls_pos_mask = pos_mask.unsqueeze(2).expand_as(pred_cls)
            if cls_pos_mask.any():
                cls_loss = cls_loss + self.bce_cls(pred_cls, cls_target)[cls_pos_mask].mean()

        norm = max(1, num_scales)
        if total_pos > 0:
            box_loss = box_loss / float(total_pos)
        total_loss = (
            self.lambda_obj * (obj_loss / norm) +
            self.lambda_cls * (cls_loss / norm) +
            self.lambda_box * box_loss
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
