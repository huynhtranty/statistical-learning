"""Hungarian Matcher cho DETR.

DETR dùng thuật toán Hungarian (phép gán tối ưu) để ghép cặp 1-1
giữa các prediction và ground truth trong mỗi ảnh.

Quá trình matching:
1. Tính cost matrix dựa trên classification cost, L1 bbox cost, và GIoU cost.
2. Dùng scipy.optimize.linear_sum_assignment (thuật toán Hungarian)
   để tìm phép gán có tổng cost nhỏ nhất.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from scipy.optimize import linear_sum_assignment

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.utils.box_ops import box_iou, generalized_box_iou, cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    """Thực hiện ghép cặp 1-1 giữa predictions và targets bằng thuật toán Hungarian.

    Cost function là tổ hợp có trọng số của:
    - Classification cost: dựa trên xác suất lớp predicted.
    - L1 cost: sai số tuyệt đối giữa toạ độ bbox.
    - GIoU cost: 1 - GIoU giữa predicted box và target box.

    Args:
        cost_class: Trọng số cho classification cost.
        cost_bbox: Trọng số cho L1 bbox cost.
        cost_giou: Trọng số cho GIoU cost.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> list[tuple[Tensor, Tensor]]:
        """Tìm phép gán tối ưu cho mỗi ảnh trong batch.

        Args:
            outputs: Dict chứa:
                - "pred_logits": (batch, num_queries, num_classes + 1)
                - "pred_boxes":  (batch, num_queries, 4) định dạng cxcywh chuẩn hoá.
            targets: Danh sách dict cho mỗi ảnh, mỗi dict chứa:
                - "labels": (num_objects,)
                - "boxes":  (num_objects, 4) định dạng cxcywh chuẩn hoá.

        Returns:
            Danh sách tuple (pred_indices, tgt_indices) cho mỗi ảnh.
            pred_indices: index các query được chọn.
            tgt_indices: index các ground truth tương ứng.
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten predictions cho toàn batch để tính cost một lần
        pred_probs = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (batch * num_queries, num_cls + 1)
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)               # (batch * num_queries, 4)

        # Ghép tất cả targets
        tgt_labels = torch.cat([t["labels"] for t in targets])
        tgt_boxes = torch.cat([t["boxes"] for t in targets])

        if tgt_labels.numel() == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
                    for _ in range(batch_size)]

        # --- Tính các thành phần cost ---

        # Classification cost: lấy xác suất tại lớp target (cost thấp = xác suất cao)
        cost_class = -pred_probs[:, tgt_labels]

        # L1 bounding box cost
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            cxcywh_to_xyxy(pred_boxes),
            cxcywh_to_xyxy(tgt_boxes),
        )

        # Tổng hợp cost matrix
        cost_matrix = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        # Reshape lại theo batch
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # Áp dụng thuật toán Hungarian cho từng ảnh
        sizes = [len(t["boxes"]) for t in targets]
        indices = []
        for i, c in enumerate(cost_matrix.split(sizes, dim=-1)):
            c_i = c[i]  # (num_queries, num_targets_i)
            pred_idx, tgt_idx = linear_sum_assignment(c_i.numpy())
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(tgt_idx, dtype=torch.long),
            ))

        return indices
