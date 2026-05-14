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

    Cải tiến so với baseline:
    - Focal Loss cho objectness (giảm dominance của vô vàn negative cells).
    - Scale-aware assignment: mỗi GT box chỉ gán vào scale phù hợp với kích thước.
    - Box loss = 1 - GIoU thay vì SmoothL1 trên (tx,ty,bw,bh) — gradient mạnh hơn.
    """

    def __init__(
        self,
        num_classes: int,
        num_anchors: int = 3,
        image_size: int = 640,
        lambda_noobj: float = 1.0,
        lambda_obj: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_box: float = 5.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.image_size = float(image_size)
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _focal_factor(logits: Tensor, targets: Tensor, alpha: float, gamma: float) -> Tensor:
        """Focal weighting factor: alpha_t * (1 - p_t)^gamma."""
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return alpha_t * (1 - p_t).pow(gamma)

    def _assign_scale_for_box(self, bw_norm: Tensor, bh_norm: Tensor, num_scales: int) -> Tensor:
        """Heuristic gán mỗi GT vào 1 scale dựa trên kích thước lớn nhất của box.
        Convention: predictions[0] = P3 (stride 8, lưới to), [-1] = P5 (stride 32, lưới nhỏ).
        - max_side < 0.15 → P3 (object nhỏ, cần lưới chi tiết để localize).
        - 0.15 ≤ max_side < 0.45 → P4 (object vừa, ~mode của dataset này).
        - max_side ≥ 0.45 → P5 (object to phủ phần lớn ảnh).

        Dùng max_side thay vì area để chống lệch đối với box dạng "thanh dài"
        (ví dụ giraffe — chiều cao lớn nhưng chiều rộng nhỏ).
        """
        max_side = torch.maximum(bw_norm, bh_norm)
        scale_idx = torch.where(
            max_side < 0.15, torch.zeros_like(max_side, dtype=torch.long),
            torch.where(max_side < 0.45, torch.ones_like(max_side, dtype=torch.long),
                        torch.full_like(max_side, num_scales - 1, dtype=torch.long))
        )
        return scale_idx.clamp(0, num_scales - 1)

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

        # Pre-compute scale assignment cho từng GT box mỗi batch image
        per_image_assignments: list[Tensor] = []
        for b in range(batch_size):
            boxes = target_boxes[b]
            if boxes.numel() == 0:
                per_image_assignments.append(torch.empty(0, dtype=torch.long, device=device))
                continue
            boxes = boxes.to(device)
            bw_norm = ((boxes[:, 2] - boxes[:, 0]) / img_size).clamp(1e-6, 1.0)
            bh_norm = ((boxes[:, 3] - boxes[:, 1]) / img_size).clamp(1e-6, 1.0)
            per_image_assignments.append(
                self._assign_scale_for_box(bw_norm, bh_norm, num_scales)
            )

        for scale_idx, pred in enumerate(predictions):
            _, channels, h, w = pred.shape
            expected = self.num_anchors * (5 + self.num_classes)
            if channels != expected:
                continue

            pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, h, w)
            pred_boxes = pred[:, :, :4, :, :]   # tx, ty, tw, th (logits)
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

                # Lọc các GT thuộc về scale hiện tại
                scale_mask = per_image_assignments[b] == scale_idx
                if scale_mask.sum() == 0:
                    continue

                boxes_s = boxes[scale_mask]
                labels_s = labels[scale_mask]

                cx = ((boxes_s[:, 0] + boxes_s[:, 2]) * 0.5 / img_size).clamp(0.0, 1.0 - 1e-6)
                cy = ((boxes_s[:, 1] + boxes_s[:, 3]) * 0.5 / img_size).clamp(0.0, 1.0 - 1e-6)
                bw = ((boxes_s[:, 2] - boxes_s[:, 0]) / img_size).clamp(1e-6, 1.0)
                bh = ((boxes_s[:, 3] - boxes_s[:, 1]) / img_size).clamp(1e-6, 1.0)

                gx = cx * w
                gy = cy * h
                gi = torch.floor(gx).long().clamp(0, w - 1)
                gj = torch.floor(gy).long().clamp(0, h - 1)
                tx = gx - gi.float()
                ty = gy - gj.float()

                valid = (labels_s >= 0) & (labels_s < self.num_classes)
                if valid.sum() == 0:
                    continue

                gi = gi[valid]
                gj = gj[valid]
                tx = tx[valid]
                ty = ty[valid]
                bw = bw[valid]
                bh = bh[valid]
                labels_v = labels_s[valid]

                # Gán positive cho tất cả anchors tại cell tương ứng.
                for a in range(self.num_anchors):
                    obj_target[b, a, gj, gi] = 1.0
                    pos_mask[b, a, gj, gi] = True
                    cls_target[b, a, labels_v, gj, gi] = 1.0

                    pred_xywh = pred_boxes[b, a, :, gj, gi].transpose(0, 1)
                    pred_xy = pred_xywh[:, :2].sigmoid()
                    pred_wh = pred_xywh[:, 2:].sigmoid()

                    # Build xyxy boxes (toạ độ chuẩn hoá [0,1] toàn ảnh) để tính GIoU.
                    # Pred center = (gi + sigmoid(tx)) / w, normalized.
                    cell_size_x = 1.0 / float(w)
                    cell_size_y = 1.0 / float(h)
                    pcx = (gi.float() + pred_xy[:, 0]) * cell_size_x
                    pcy = (gj.float() + pred_xy[:, 1]) * cell_size_y
                    pw = pred_wh[:, 0]
                    ph = pred_wh[:, 1]
                    pred_xyxy = torch.stack(
                        (pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2),
                        dim=1,
                    )
                    tcx = cx[valid]
                    tcy = cy[valid]
                    tgt_xyxy = torch.stack(
                        (tcx - bw / 2, tcy - bh / 2, tcx + bw / 2, tcy + bh / 2),
                        dim=1,
                    )
                    giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
                    # Lấy đường chéo (mỗi pred khớp đúng GT cùng index)
                    giou_diag = giou.diagonal()
                    box_loss = box_loss + (1.0 - giou_diag).sum()

                total_pos += int(gi.numel()) * self.num_anchors

            # ---- Objectness loss với Focal weighting ----
            obj_bce = self.bce_obj(pred_obj, obj_target)
            focal_w = self._focal_factor(
                pred_obj.detach(), obj_target, self.focal_alpha, self.focal_gamma
            )
            obj_elementwise = obj_bce * focal_w

            pos_count = pos_mask.sum().item()
            neg_mask = ~pos_mask
            neg_count = neg_mask.sum().item()

            if pos_count > 0:
                obj_pos = obj_elementwise[pos_mask].sum() / max(pos_count, 1)
            else:
                obj_pos = torch.zeros((), device=device)

            if neg_count > 0:
                # Chia cho pos_count để negatives được scale theo số positives (focal-loss convention).
                obj_neg = obj_elementwise[neg_mask].sum() / max(pos_count, 1)
            else:
                obj_neg = torch.zeros((), device=device)

            obj_loss = obj_loss + self.lambda_obj * obj_pos + self.lambda_noobj * obj_neg

            # ---- Classification loss (chỉ ở positive cells) ----
            cls_pos_mask = pos_mask.unsqueeze(2).expand_as(pred_cls)
            if cls_pos_mask.any():
                cls_bce = self.bce_cls(pred_cls, cls_target)[cls_pos_mask]
                cls_loss = cls_loss + cls_bce.mean()

        norm_pos = max(1, total_pos)
        box_loss = box_loss / float(norm_pos)
        total_loss = (
            obj_loss / max(1, num_scales) +
            self.lambda_cls * (cls_loss / max(1, num_scales)) +
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
