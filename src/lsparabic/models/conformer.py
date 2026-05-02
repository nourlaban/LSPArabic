from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lsparabic.models.interfaces import BaseSequenceModel


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConformerMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        y = self.norm(x)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask)
        return self.dropout(y)


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Linear(d_model, d_model * 2)
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.pointwise_in(y)                      # (B, T, 2D)
        y, gate = y.chunk(2, dim=-1)                  # gated linear unit
        y = y * torch.sigmoid(gate)
        y = y.transpose(1, 2)                         # (B, D, T) for Conv1d
        y = self.activation(self.batch_norm(self.depthwise(y)))
        y = y.transpose(1, 2)                         # (B, T, D)
        return self.pointwise_out(y)


class ConformerBlock(nn.Module):
    """Macaron-style Conformer: FF/2 → MHSA → Conv → FF/2."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff1 = ConformerFeedForward(d_model, ffn_expansion, dropout)
        self.mhsa = ConformerMultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConformerConvolution(d_model, conv_kernel_size)
        self.ff2 = ConformerFeedForward(d_model, ffn_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ConformerEncoder(BaseSequenceModel):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ffn_expansion, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """x: (B, T, input_dim) -> (B, T, d_model)"""
        x = self.input_proj(x)
        key_padding_mask = self._lengths_to_mask(lengths, x.size(1))
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x

    @property
    def output_dim(self) -> int:
        return self._d_model

    @staticmethod
    def _lengths_to_mask(lengths: Tensor, max_len: int) -> Tensor:
        """True where position should be *ignored* (padding)."""
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask
