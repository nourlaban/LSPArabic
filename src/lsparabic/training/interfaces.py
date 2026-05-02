from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseLossComponent(ABC, nn.Module):
    @abstractmethod
    def forward(self, predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
        """Compute scalar loss from predictions and targets dicts."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for logging."""

    @property
    @abstractmethod
    def weight(self) -> float:
        """Scalar multiplier applied in total loss."""
