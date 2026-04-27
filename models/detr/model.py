"""Kiến trúc DETR (DEtection TRansformer) cho object detection.

DETR loại bỏ các thành phần thủ công như anchor và NMS bằng cách
sử dụng transformer và bipartite matching:

1. CNN backbone trích xuất feature map.
2. Transformer encoder xử lý feature + positional encoding.
3. Transformer decoder dùng object queries để dự đoán set of objects.
4. Classification head và bbox head dự đoán lớp và toạ độ box.

Mỗi object query dự đoán một vật thể hoặc "no object" (∅).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import DETRBackbone
from .transformer import DETRTransformer


class DETR(nn.Module):
    """Mô hình DETR hoàn chỉnh.

    Args:
        num_classes: Số lớp vật thể (không tính "no object").
        hidden_dim: Số chiều ẩn của transformer.
        num_queries: Số object queries (số vật thể tối đa có thể detect mỗi ảnh).
        num_heads: Số attention heads trong transformer.
        num_encoder_layers: Số lớp encoder.
        num_decoder_layers: Số lớp decoder.
        pretrained_backbone: Dùng backbone pretrained hay không.
    """

    def __init__(
        self,
        num_classes: int = 5,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Backbone CNN
        self.backbone = DETRBackbone(
            hidden_dim=hidden_dim,
            pretrained=pretrained_backbone,
        )

        # Transformer encoder-decoder
        self.transformer = DETRTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        # Object queries: mỗi query học cách "hỏi" về một vật thể
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Classification head: dự đoán lớp cho mỗi query
        # num_classes + 1 vì có thêm lớp "no object" (∅)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)

        # Bounding box head: dự đoán toạ độ box (cx, cy, w, h) chuẩn hoá [0, 1]
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        # TODO: Thêm auxiliary loss (dự đoán ở mỗi decoder layer, không chỉ layer cuối).
        # TODO: Khởi tạo trọng số cho class_head và bbox_head theo cách đặc biệt.

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass qua toàn bộ pipeline DETR.

        Args:
            x: Ảnh đầu vào (batch, 3, H, W).

        Returns:
            Dict chứa:
                - "pred_logits": (batch, num_queries, num_classes + 1)
                - "pred_boxes":  (batch, num_queries, 4) định dạng cxcywh chuẩn hoá.
        """
        # Trích xuất feature từ backbone
        features = self.backbone(x)  # (batch, hidden_dim, H/32, W/32)

        # Transformer encoder-decoder
        query_embed = self.query_embed.weight  # (num_queries, hidden_dim)
        decoder_output = self.transformer(features, query_embed)
        # decoder_output: (num_queries, batch, hidden_dim)

        # Chuyển về (batch, num_queries, hidden_dim)
        decoder_output = decoder_output.permute(1, 0, 2)

        # Dự đoán lớp và bounding box
        pred_logits = self.class_head(decoder_output)       # (batch, num_queries, num_classes + 1)
        pred_boxes = self.bbox_head(decoder_output).sigmoid()  # (batch, num_queries, 4) chuẩn hoá [0, 1]

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }


def build_detr(
    num_classes: int = 5,
    num_queries: int = 100,
    hidden_dim: int = 256,
    pretrained_backbone: bool = True,
) -> DETR:
    """Tạo mô hình DETR.

    Args:
        num_classes: Số lớp vật thể (không tính "no object").
        num_queries: Số object queries (vật thể tối đa mỗi ảnh).
        hidden_dim: Số chiều ẩn của transformer.
        pretrained_backbone: Dùng backbone pretrained trên ImageNet hay không.

    Returns:
        Mô hình DETR sẵn sàng để train hoặc inference.
    """
    # TODO: Cho phép cấu hình chi tiết hơn (num_heads, num_layers, v.v.) từ config.yaml.
    # TODO: Hỗ trợ load checkpoint pretrained trên COCO từ HuggingFace.
    return DETR(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        pretrained_backbone=pretrained_backbone,
    )
