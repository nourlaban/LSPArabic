from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lsparabic.training.interfaces import BaseLossComponent


class CTCLossWrapper(BaseLossComponent):
    def __init__(
        self,
        blank_idx: int = 0,
        weight: float = 1.0,
        zero_infinity: bool = True,
    ) -> None:
        super().__init__()
        self._weight = weight
        self._name = "ctc_loss"
        self.ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=zero_infinity, reduction="mean")

    def forward(self, predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
        """
        predictions["logits"]:       (B, T, vocab+1) log-probs
        predictions["lengths"]:      (B,)
        targets["labels"]:           (B, L) token ids
        targets["label_lengths"]:    (B,)
        """
        logits = predictions["logits"]        # (B, T, V)
        log_probs = logits.permute(1, 0, 2)  # (T, B, V) required by CTCLoss
        input_lengths = predictions["lengths"]
        targets_flat = targets["labels"]
        target_lengths = targets["label_lengths"]

        return self.ctc(log_probs, targets_flat, input_lengths, target_lengths)

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight
