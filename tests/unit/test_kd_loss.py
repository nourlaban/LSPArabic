from __future__ import annotations

import torch
import pytest

from lsparabic.training.kd_loss import KnowledgeDistillationLoss


def test_loss_non_negative():
    loss_fn = KnowledgeDistillationLoss(temperature=4.0)
    B, T, V = 2, 10, 100
    student_logits = torch.randn(B, T, V).log_softmax(-1)
    teacher_logits = torch.randn(B, T, V)
    preds = {"logits": student_logits}
    tgts = {"teacher_logits": teacher_logits}
    loss = loss_fn(preds, tgts)
    assert loss.item() >= 0.0


def test_loss_zero_when_student_equals_teacher():
    """When student and teacher have identical distributions, KL divergence ≈ 0."""
    loss_fn = KnowledgeDistillationLoss(temperature=1.0)
    B, T, V = 2, 5, 50
    logits = torch.randn(B, T, V)
    preds = {"logits": logits.log_softmax(-1)}
    tgts = {"teacher_logits": logits}
    loss = loss_fn(preds, tgts)
    assert loss.item() < 1e-4


def test_weight_property():
    loss_fn = KnowledgeDistillationLoss(weight=0.7)
    assert loss_fn.weight == pytest.approx(0.7)
