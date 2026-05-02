from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lsparabic.models.interfaces import BaseSequenceModel


class MambaResidualBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError(
                "mamba_ssm is required for MambaSSMEncoder. "
                "Install with: pip install 'lsparabic[mamba]'"
            ) from e
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mamba(self.norm(x))


class MambaSSMEncoder(BaseSequenceModel):
    """Drop-in Mamba-based replacement for ConformerEncoder.

    Requires the optional mamba_ssm package (CUDA only).
    Install: pip install 'lsparabic[mamba]'
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        num_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.layers = nn.ModuleList([
            MambaResidualBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """x: (B, T, input_dim) -> (B, T, d_model)"""
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    @property
    def output_dim(self) -> int:
        return self._d_model
