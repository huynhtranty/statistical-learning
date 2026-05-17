"""Transformer encoder-decoder và positional encoding cho DETR.

DETR sử dụng transformer tiêu chuẩn:
- Encoder nhận feature map từ backbone (flatten thành sequence) + positional encoding.
- Decoder nhận object queries + output của encoder, dự đoán set of objects.

Positional encoding dùng kiểu sine/cosine 2D để mã hoá vị trí không gian
của từng pixel trong feature map.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding2D(nn.Module):
    """Positional encoding dạng sine/cosine cho feature map 2D.

    Mã hoá vị trí (x, y) của từng pixel trong feature map bằng
    hàm sin/cos ở các tần số khác nhau, giúp transformer biết
    thông tin vị trí không gian.

    Args:
        hidden_dim: Số chiều ẩn của transformer.
            Nửa đầu dùng cho trục y, nửa sau dùng cho trục x.
        temperature: Hằng số nhiệt độ điều chỉnh tần số (mặc định 10000).
    """

    def __init__(self, hidden_dim: int = 256, temperature: float = 10000.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        """Tạo positional encoding cho feature map.

        Args:
            x: Feature map (batch, hidden_dim, H, W) — chỉ dùng shape, không dùng giá trị.

        Returns:
            Positional encoding (1, hidden_dim, H, W) có thể cộng trực tiếp vào feature map.
        """
        _, _, h, w = x.shape
        half_dim = self.hidden_dim // 2

        # Tạo toạ độ y và x chuẩn hoá về [0, 1]
        y_pos = torch.arange(h, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, w) / h
        x_pos = torch.arange(w, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(h, 1) / w

        # Tạo các tần số cho sin/cos
        dim_t = torch.arange(half_dim, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half_dim)

        # Encoding cho trục y
        pe_y = y_pos.unsqueeze(-1) / dim_t  # (H, W, half_dim)
        pe_y = torch.stack([pe_y[..., 0::2].sin(), pe_y[..., 1::2].cos()], dim=-1)
        pe_y = pe_y.flatten(-2)  # (H, W, half_dim)

        # Encoding cho trục x
        pe_x = x_pos.unsqueeze(-1) / dim_t  # (H, W, half_dim)
        pe_x = torch.stack([pe_x[..., 0::2].sin(), pe_x[..., 1::2].cos()], dim=-1)
        pe_x = pe_x.flatten(-2)  # (H, W, half_dim)

        # Ghép encoding y và x
        pe = torch.cat([pe_y, pe_x], dim=-1)  # (H, W, hidden_dim)
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # (1, hidden_dim, H, W)

        return pe


class DETRTransformer(nn.Module):
    """Transformer encoder-decoder cho DETR.

    - Encoder: xử lý chuỗi feature từ backbone kèm positional encoding.
    - Decoder: dùng object queries để "hỏi" encoder và dự đoán set of objects.

    Args:
        hidden_dim: Số chiều ẩn (d_model) của transformer.
        num_heads: Số attention heads.
        num_encoder_layers: Số lớp encoder.
        num_decoder_layers: Số lớp decoder.
        dim_feedforward: Số chiều trong FFN (feed-forward network) bên trong mỗi layer.
        dropout: Tỷ lệ dropout.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.pos_encoder = PositionalEncoding2D(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # TODO: Thêm layer norm cuối cho encoder/decoder nếu cần.
        # TODO: Khởi tạo trọng số theo Xavier uniform như paper gốc.

    def forward(self, src: Tensor, query_embed: Tensor) -> Tensor:
        """Forward qua transformer encoder-decoder.

        Args:
            src: Feature map từ backbone (batch, hidden_dim, H, W).
            query_embed: Object query embeddings (num_queries, hidden_dim).

        Returns:
            Output của decoder: (num_queries, batch, hidden_dim).
        """
        batch_size = src.shape[0]

        # Tạo positional encoding cho feature map
        pos = self.pos_encoder(src)  # (1, hidden_dim, H, W)

        # Flatten feature map: (batch, hidden_dim, H, W) → (H*W, batch, hidden_dim)
        src_flat = src.flatten(2).permute(2, 0, 1)   # (H*W, batch, hidden_dim)
        pos_flat = pos.flatten(2).permute(2, 0, 1)   # (H*W, 1, hidden_dim)

        # Cộng positional encoding vào source
        src_with_pos = src_flat + pos_flat

        # Encoder
        memory = self.encoder(src_with_pos)  # (H*W, batch, hidden_dim)

        # Chuẩn bị object queries cho decoder
        # query_embed: (num_queries, hidden_dim) → (num_queries, batch, hidden_dim)
        tgt = torch.zeros_like(query_embed.unsqueeze(1).repeat(1, batch_size, 1))
        query_pos = query_embed.unsqueeze(1).repeat(1, batch_size, 1)

        # Decoder: object queries attend vào memory từ encoder
        output = self.decoder(tgt + query_pos, memory)  # (num_queries, batch, hidden_dim)

        return output
