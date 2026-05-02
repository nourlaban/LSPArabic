from __future__ import annotations

import torch
import pytest

from lsparabic.training.feature_regression_loss import FeatureRegressionLoss


def test_loss_non_negative():
    loss_fn = FeatureRegressionLoss(student_dim=64, teacher_dim=128)
    B, T = 2, 10
    preds = {"encoder_out": torch.randn(B, T, 64)}
    tgts = {"teacher_hidden": torch.randn(B, T, 128)}
    loss = loss_fn(preds, tgts)
    assert loss.item() >= 0.0


def test_weight_property():
    loss_fn = FeatureRegressionLoss(student_dim=64, teacher_dim=128, weight=0.1)
    assert loss_fn.weight == pytest.approx(0.1)


def test_name_property():
    loss_fn = FeatureRegressionLoss(student_dim=64, teacher_dim=64)
    assert loss_fn.name == "feature_regression_loss"
