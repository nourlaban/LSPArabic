from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lsparabic.training.interfaces import BaseLossComponent


class KnowledgeDistillationLoss(BaseLossComponent):
    """KL divergence between student and teacher soft targets (temperature-scaled).

    L_kd = T^2 * KL(softmax(teacher/T) || log_softmax(student/T))
    """

    def __init__(self, temperature: float = 4.0, weight: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self._weight = weight
        self._name = "kd_loss"

    def forward(self, predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
        """
        predictions["logits"]:       (B, T_s, V_student)  log-probs
        targets["teacher_logits"]:   (B, T_t, V_teacher)  raw logits from teacher
        Sequence lengths may differ; we pool both to (B, V) via mean before KLD.
        """
        student_logits = predictions["logits"]       # (B, T_s, V_s) log-probs
        teacher_logits = targets["teacher_logits"]   # (B, T_t, V_t) raw logits

        # Mean-pool over time to get per-sample distributions
        student_dist = student_logits.mean(dim=1)    # (B, V_s)
        teacher_dist = teacher_logits.mean(dim=1)    # (B, V_t)

        # If vocab sizes differ, project teacher to student vocab size
        if teacher_dist.size(-1) != student_dist.size(-1):
            teacher_dist = teacher_dist[..., : student_dist.size(-1)]

        T = self.temperature
        soft_student = F.log_softmax(student_dist / T, dim=-1)
        soft_teacher = F.softmax(teacher_dist / T, dim=-1)

        loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
        return loss

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight
