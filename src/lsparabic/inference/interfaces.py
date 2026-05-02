from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class BaseDecoder(ABC):
    @abstractmethod
    def decode(self, logits: Tensor, lengths: Tensor) -> list[str]:
        """Decode CTC log-prob tensor to list of text strings."""
