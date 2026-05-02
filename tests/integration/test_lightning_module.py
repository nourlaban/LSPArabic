"""Integration test: VSRLightningModule runs a training + validation step."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf


def _build_module(tiny_tokenizer):
    from lsparabic.models.conformer import ConformerEncoder
    from lsparabic.models.decoder import CTCDecoder
    from lsparabic.models.visual_encoder import ResNet3D
    from lsparabic.models.vsr_model import VSRModel
    from lsparabic.training.lightning_module import VSRLightningModule

    vocab = tiny_tokenizer.vocab_size
    visual_encoder = ResNet3D(output_dim=64)
    sequence_model = ConformerEncoder(input_dim=64, d_model=64, num_layers=1)
    decoder = CTCDecoder(input_dim=64, vocab_size=vocab)
    model = VSRModel(visual_encoder, sequence_model, decoder, proj_dim=64)

    teacher = MagicMock()
    teacher.return_value = {
        "logits": torch.zeros(1, 10, 51865),
        "hidden_states": torch.zeros(1, 10, 64),
    }

    cfg = OmegaConf.create({
        "training": {
            "loss_weights": {"ctc": 1.0, "kd": 0.5, "feature_regression": 0.1},
            "kd": {"temperature": 4.0},
            "feature_regression": {"student_dim": 64, "teacher_dim": 64},
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "betas": [0.9, 0.98],
            "eps": 1e-9,
        },
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 10,
            "eta_min": 1e-6,
            "warmup_epochs": 0,
        },
    })

    return VSRLightningModule(cfg=cfg, model=model, teacher=teacher, tokenizer=tiny_tokenizer)


def _make_batch(B: int = 2, T: int = 20, vocab: int = 50):
    return {
        "frames": torch.randn(B, T, 3, 96, 96),
        "frame_lengths": torch.tensor([T, T // 2]),
        "labels": torch.zeros(B, 5, dtype=torch.long),
        "label_lengths": torch.tensor([4, 3]),
        "video_ids": ["v1", "v2"],
    }


def test_training_step_no_crash(tiny_tokenizer):
    module = _build_module(tiny_tokenizer)
    batch = _make_batch()
    loss = module.training_step(batch, 0)
    assert loss.item() == loss.item()  # not NaN


def test_validation_step_no_crash(tiny_tokenizer):
    module = _build_module(tiny_tokenizer)
    batch = _make_batch()
    module.validation_step(batch, 0)
    metrics = module._val_metrics.compute()
    assert "wer" in metrics


def test_configure_optimizers(tiny_tokenizer):
    module = _build_module(tiny_tokenizer)
    result = module.configure_optimizers()
    assert "optimizer" in result
    assert "lr_scheduler" in result
