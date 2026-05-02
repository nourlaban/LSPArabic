from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lsparabic.training.interfaces import BaseLossComponent


class FeatureProjection(nn.Module):
    """Learnable linear map to align student and teacher hidden dimensions."""

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class FeatureRegressionLoss(BaseLossComponent):
    """MSE between projected student intermediate features and teacher hidden states.

    Encourages the student (video stream) to learn representations aligned
    with the teacher's (audio) internal representations.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = FeatureProjection(student_dim, teacher_dim)
        self._weight = weight
        self._name = "feature_regression_loss"

    def forward(self, predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
        """
        predictions["encoder_out"]: (B, T_s, student_dim)
        targets["teacher_hidden"]:  (B, T_t, teacher_dim)
        Both are mean-pooled over time before MSE.
        """
        student_feat = predictions["encoder_out"].mean(dim=1)   # (B, student_dim)
        teacher_feat = targets["teacher_hidden"].mean(dim=1)    # (B, teacher_dim)

        student_proj = self.projection(student_feat)             # (B, teacher_dim)
        return F.mse_loss(student_proj, teacher_feat.detach())

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight
