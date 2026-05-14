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

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
import yaml

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


class HuggingFaceDETR(nn.Module):
    """Wrapper quanh HuggingFace `facebook/detr-resnet-50` (COCO-pretrained).

    Mở interface giống DETR custom: forward(x) → {"pred_logits", "pred_boxes"}.
    Tự normalize ImageNet bên trong nên dataset pipeline không cần thay đổi
    (ảnh đầu vào vẫn là tensor [0,1] sau khi `F.to_tensor`).
    """

    # ImageNet mean/std (cùng với mọi backbone torchvision).
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, num_classes: int = 10, num_queries: int = 100) -> None:
        super().__init__()
        from transformers import DetrForObjectDetection

        # ignore_mismatched_sizes=True → drop COCO 91-class head, init head mới
        # với num_labels=num_classes; phần backbone + transformer giữ nguyên COCO weights.
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            num_queries=num_queries,
            ignore_mismatched_sizes=True,
        )
        self.num_classes = num_classes
        self.num_queries = num_queries

        mean = torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std", std)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # Ảnh đầu vào [0,1] → ImageNet-normalized cho HF DETR.
        x = (x - self.img_mean) / self.img_std
        out = self.model(pixel_values=x)
        return {
            "pred_logits": out.logits,     # (B, Q, num_classes+1)
            "pred_boxes": out.pred_boxes,  # (B, Q, 4) cxcywh normalized [0,1]
        }


def build_detr(
    num_classes: int | None = None,
    num_queries: int | None = None,
    hidden_dim: int | None = None,
    num_heads: int | None = None,
    num_encoder_layers: int | None = None,
    num_decoder_layers: int | None = None,
    pretrained_backbone: bool | None = None,
    pretrained_coco: bool = True,
) -> nn.Module:
    """Tạo mô hình DETR.

    Args:
        pretrained_coco: Nếu True (mặc định), dùng HuggingFace
            `facebook/detr-resnet-50` COCO-pretrained, chỉ thay class head.
            Nếu False, build DETR custom từ scratch (chỉ ImageNet backbone).
        num_classes: Số lớp foreground (không tính "no object").
        num_queries: Số object queries.
        Các tham số khác chỉ áp dụng khi pretrained_coco=False.

    Returns:
        Mô hình DETR (custom hoặc HF wrapper) với interface
        forward(x) → {"pred_logits", "pred_boxes"}.
    """
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f).get("model", {})

    num_classes = model_cfg.get("num_classes", 10) if num_classes is None else num_classes
    num_queries = model_cfg.get("num_queries", 100) if num_queries is None else num_queries

    if pretrained_coco:
        return HuggingFaceDETR(num_classes=num_classes, num_queries=num_queries)

    # Custom from-scratch DETR (chỉ ImageNet backbone).
    hidden_dim = model_cfg.get("hidden_dim", 256) if hidden_dim is None else hidden_dim
    num_heads = model_cfg.get("num_heads", 8) if num_heads is None else num_heads
    num_encoder_layers = (
        model_cfg.get("num_encoder_layers", 6) if num_encoder_layers is None else num_encoder_layers
    )
    num_decoder_layers = (
        model_cfg.get("num_decoder_layers", 6) if num_decoder_layers is None else num_decoder_layers
    )
    pretrained_backbone = (
        model_cfg.get("pretrained_backbone", True)
        if pretrained_backbone is None
        else pretrained_backbone
    )

    return DETR(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        pretrained_backbone=pretrained_backbone,
    )
