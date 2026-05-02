from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseVisualEncoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C, H, W) -> (B, T, D)"""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the encoder output features."""


class BaseSequenceModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """x: (B, T, D) -> (B, T, D)"""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the sequence model output."""
