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

# Anchor priors (normalized 0..1, w h) cho 3 scale × 3 anchor.
# Lấy cảm hứng từ YOLOv5 anchors (pixel @ 640) chia cho 640:
#   P3/8:  10,13   16,30   33,23
#   P4/16: 30,61   62,45   59,119
#   P5/32: 116,90  156,198 373,326
DEFAULT_YOLO_ANCHORS: list[list[tuple[float, float]]] = [
    [(0.0156, 0.0203), (0.0250, 0.0469), (0.0516, 0.0359)],   # P3 (small)
    [(0.0469, 0.0953), (0.0969, 0.0703), (0.0922, 0.1859)],   # P4 (medium)
    [(0.1813, 0.1406), (0.2438, 0.3094), (0.5828, 0.5094)],   # P5 (large)
]


class YOLOLoss(nn.Module):
    """Loss cho kiến trúc YOLO-style với anchor-aware matching.

    Cải tiến so với baseline:
    - Anchor priors per scale × per anchor: mỗi GT chỉ assigned cho 1 anchor
      có shape IoU lớn nhất (top-1) → 3 anchors phân hoá tốt, không collapse.
    - Center-region assignment (YOLOv5 style): mỗi GT gán cho 1-3 cell (center +
      neighbor cùng nửa cell) → tăng số positive samples 1.5-3x.
    - YOLOv5 xy parameterization: pred_xy = 2*sigmoid(tx) - 0.5, range [-0.5, 1.5].
    - YOLOv5 wh parameterization: pred_wh = (2*sigmoid(t))^2, fraction of image.
    - Focal Loss cho objectness (giảm dominance của negative cells).
    - Box loss = (1 - GIoU) + L1 trên (cx, cy, w, h) chuẩn hoá.
    """

    def __init__(
        self,
        num_classes: int,
        num_anchors: int = 3,
        image_size: int = 640,
        anchors: list[list[tuple[float, float]]] | None = None,
        lambda_noobj: float = 1.0,
        lambda_obj: float = 1.0,
        lambda_cls: float = 1.0,
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

        anchors = anchors or DEFAULT_YOLO_ANCHORS
        if any(len(scale_anchors) != num_anchors for scale_anchors in anchors):
            raise ValueError(
                f"Mỗi scale phải có đúng {num_anchors} anchors, "
                f"got: {[len(a) for a in anchors]}"
            )
        self.num_scales = len(anchors)
        # Stack thành 1 tensor (num_scales, num_anchors, 2) để vectorise.
        anchors_tensor = torch.tensor(anchors, dtype=torch.float32)  # (S, A, 2)
        self.register_buffer("anchors", anchors_tensor)

        self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _focal_factor(logits: Tensor, targets: Tensor, alpha: float, gamma: float) -> Tensor:
        """Focal weighting factor: alpha_t * (1 - p_t)^gamma."""
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return alpha_t * (1 - p_t).pow(gamma)

    @staticmethod
    def _shape_iou(box_wh: Tensor, anchor_wh: Tensor) -> Tensor:
        """Shape IoU centered at origin giữa GT boxes và anchor priors.

        Args:
            box_wh: (N, 2) tensor (w, h) normalized [0, 1].
            anchor_wh: (M, 2) tensor (w, h) normalized [0, 1].

        Returns:
            (N, M) shape IoU. Đo độ tương thích về kích thước/aspect — center bằng 0.
        """
        inter_w = torch.minimum(box_wh[:, None, 0], anchor_wh[None, :, 0])
        inter_h = torch.minimum(box_wh[:, None, 1], anchor_wh[None, :, 1])
        inter = inter_w * inter_h
        area_box = (box_wh[:, 0] * box_wh[:, 1]).unsqueeze(1)  # (N, 1)
        area_anc = (anchor_wh[:, 0] * anchor_wh[:, 1]).unsqueeze(0)  # (1, M)
        return inter / (area_box + area_anc - inter + 1e-9)

    def _match_gt_to_anchors(
        self, box_wh: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Cho mỗi GT, chọn (scale_idx, anchor_idx) có shape IoU cao nhất.

        Args:
            box_wh: (N, 2) tensor (w_norm, h_norm).

        Returns:
            (scale_idx, anchor_idx) — mỗi cái là (N,) long tensor.
        """
        # anchors: (S, A, 2) → flatten sang (S*A, 2). Đảm bảo cùng device với input
        # vì YOLOLoss có thể không được .to(device) (chỉ model được move).
        all_anchors = self.anchors.view(-1, 2).to(box_wh.device)  # (S*A, 2)
        ious = self._shape_iou(box_wh, all_anchors)  # (N, S*A)
        best = ious.argmax(dim=1)  # (N,)
        scale_idx = best // self.num_anchors
        anchor_idx = best % self.num_anchors
        return scale_idx, anchor_idx

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

        if num_scales != self.num_scales:
            raise ValueError(
                f"predictions có {num_scales} scale nhưng anchors chỉ định "
                f"{self.num_scales} scale."
            )

        obj_loss = torch.zeros((), device=device)
        cls_loss = torch.zeros((), device=device)
        box_loss = torch.zeros((), device=device)
        total_pos = 0

        # Pre-compute matched (scale_idx, anchor_idx) cho từng GT mỗi batch image.
        per_image_scale: list[Tensor] = []
        per_image_anchor: list[Tensor] = []
        for b in range(batch_size):
            boxes = target_boxes[b]
            if boxes.numel() == 0:
                per_image_scale.append(torch.empty(0, dtype=torch.long, device=device))
                per_image_anchor.append(torch.empty(0, dtype=torch.long, device=device))
                continue
            boxes = boxes.to(device)
            bw_norm = ((boxes[:, 2] - boxes[:, 0]) / img_size).clamp(1e-6, 1.0)
            bh_norm = ((boxes[:, 3] - boxes[:, 1]) / img_size).clamp(1e-6, 1.0)
            box_wh = torch.stack([bw_norm, bh_norm], dim=1)  # (N, 2)
            scale_idx, anchor_idx = self._match_gt_to_anchors(box_wh)
            per_image_scale.append(scale_idx)
            per_image_anchor.append(anchor_idx)

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

                # Lọc GT có scale match = scale hiện tại
                scale_mask = per_image_scale[b] == scale_idx
                if scale_mask.sum() == 0:
                    continue

                boxes_s = boxes[scale_mask]
                labels_s = labels[scale_mask]
                anchor_idx_s = per_image_anchor[b][scale_mask]  # (N_s,)

                cx = ((boxes_s[:, 0] + boxes_s[:, 2]) * 0.5 / img_size).clamp(0.0, 1.0 - 1e-6)
                cy = ((boxes_s[:, 1] + boxes_s[:, 3]) * 0.5 / img_size).clamp(0.0, 1.0 - 1e-6)
                bw = ((boxes_s[:, 2] - boxes_s[:, 0]) / img_size).clamp(1e-6, 1.0)
                bh = ((boxes_s[:, 3] - boxes_s[:, 1]) / img_size).clamp(1e-6, 1.0)

                valid = (labels_s >= 0) & (labels_s < self.num_classes)
                if valid.sum() == 0:
                    continue

                cx = cx[valid]
                cy = cy[valid]
                bw = bw[valid]
                bh = bh[valid]
                labels_v = labels_s[valid]
                anchor_idx_v = anchor_idx_s[valid]  # (N_v,) — anchor index của từng GT

                # ========= Center-region assignment (YOLOv5 style) =========
                # Mỗi GT gán cho 1-3 cell tại anchor đã matched.
                gx = cx * w  # GT center theo đơn vị cell
                gy = cy * h
                gi_c = torch.floor(gx).long().clamp(0, w - 1)
                gj_c = torch.floor(gy).long().clamp(0, h - 1)

                gx_frac = gx - gi_c.float()
                gy_frac = gy - gj_c.float()

                cand_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

                all_gi: list[Tensor] = []
                all_gj: list[Tensor] = []
                all_anchor: list[Tensor] = []
                all_bw: list[Tensor] = []
                all_bh: list[Tensor] = []
                all_cx: list[Tensor] = []
                all_cy: list[Tensor] = []
                all_labels: list[Tensor] = []

                for di, dj in cand_offsets:
                    if di == 0 and dj == 0:
                        keep = torch.ones_like(gx_frac, dtype=torch.bool)
                    elif di == -1:
                        keep = gx_frac < 0.5
                    elif di == 1:
                        keep = gx_frac >= 0.5
                    elif dj == -1:
                        keep = gy_frac < 0.5
                    elif dj == 1:
                        keep = gy_frac >= 0.5
                    else:
                        keep = torch.ones_like(gx_frac, dtype=torch.bool)

                    if keep.sum() == 0:
                        continue

                    ngi = (gi_c + di).clamp(0, w - 1)
                    ngj = (gj_c + dj).clamp(0, h - 1)
                    in_bounds = ((gi_c[keep] + di) >= 0) & ((gi_c[keep] + di) < w) & \
                                ((gj_c[keep] + dj) >= 0) & ((gj_c[keep] + dj) < h)
                    keep_idx = torch.where(keep)[0][in_bounds]
                    if keep_idx.numel() == 0:
                        continue

                    all_gi.append(ngi[keep_idx])
                    all_gj.append(ngj[keep_idx])
                    all_anchor.append(anchor_idx_v[keep_idx])
                    all_bw.append(bw[keep_idx])
                    all_bh.append(bh[keep_idx])
                    all_cx.append(cx[keep_idx])
                    all_cy.append(cy[keep_idx])
                    all_labels.append(labels_v[keep_idx])

                if not all_gi:
                    continue

                p_gi = torch.cat(all_gi)
                p_gj = torch.cat(all_gj)
                p_anchor = torch.cat(all_anchor)
                p_bw = torch.cat(all_bw)
                p_bh = torch.cat(all_bh)
                p_cx = torch.cat(all_cx)
                p_cy = torch.cat(all_cy)
                p_labels = torch.cat(all_labels)

                # ========= Gán positive CHỈ tại anchor đã matched =========
                cell_size_x = 1.0 / float(w)
                cell_size_y = 1.0 / float(h)

                # Vì advanced indexing với cùng cell (b, a, gj, gi) có thể trùng
                # giữa các GT, chỉ giữ unique key để tránh ghi đè im lặng.
                # Trùng key thường hiếm (chỉ khi 2 GT đè nhau hoàn toàn).
                obj_target[b, p_anchor, p_gj, p_gi] = 1.0
                pos_mask[b, p_anchor, p_gj, p_gi] = True
                cls_target[b, p_anchor, p_labels, p_gj, p_gi] = 1.0

                # Compute box prediction tại (anchor, gj, gi)
                # pred_boxes shape: (B, A, 4, H, W) → lấy (B, anchor, :, gj, gi)
                pred_xywh = pred_boxes[b, p_anchor, :, p_gj, p_gi]  # (N, 4)

                # YOLOv5 parameterization
                pred_xy = 2.0 * pred_xywh[:, :2].sigmoid() - 0.5
                pred_wh = (2.0 * pred_xywh[:, 2:].sigmoid()).pow(2)

                pcx = (p_gi.float() + pred_xy[:, 0]) * cell_size_x
                pcy = (p_gj.float() + pred_xy[:, 1]) * cell_size_y
                pw = pred_wh[:, 0].clamp(min=1e-6, max=4.0)
                ph = pred_wh[:, 1].clamp(min=1e-6, max=4.0)

                pred_xyxy = torch.stack(
                    (pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2),
                    dim=1,
                )
                tgt_xyxy = torch.stack(
                    (p_cx - p_bw / 2, p_cy - p_bh / 2,
                     p_cx + p_bw / 2, p_cy + p_bh / 2),
                    dim=1,
                )
                giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
                giou_diag = giou.diagonal()

                pred_cxcywh = torch.stack((pcx, pcy, pw, ph), dim=1)
                tgt_cxcywh = torch.stack((p_cx, p_cy, p_bw, p_bh), dim=1)
                l1_per_box = (pred_cxcywh - tgt_cxcywh).abs().sum(dim=1)

                box_loss = box_loss + (1.0 - giou_diag).sum() + l1_per_box.sum()
                total_pos += int(p_gi.numel())

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
                # Chia cho neg_count để obj_neg là trung bình per-element thực sự,
                # tránh blow-up khi ảnh ít object (pos_count nhỏ vs neg_count lớn).
                obj_neg = obj_elementwise[neg_mask].sum() / max(neg_count, 1)
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
        giou = generalized_box_iou(
            cxcywh_to_xyxy(src_boxes),
            cxcywh_to_xyxy(target_boxes)
        )
        giou_diag = torch.diag(giou)
        # Clamp to [-1, 1] range (GIoU can be -1 for completely non-overlapping boxes)
        loss_giou = (1.0 - giou_diag.clamp(-1.0, 1.0)).mean()

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
