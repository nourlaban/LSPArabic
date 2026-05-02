"""Integration test: ArabicVSRDataModule yields correctly shaped batches."""

from __future__ import annotations

import pytest
import torch


def test_datamodule_yields_correct_shapes(tiny_split_csv, tiny_tokenizer):
    splits_dir, roi_dir = tiny_split_csv
    from omegaconf import OmegaConf
    from lsparabic.data.datamodule import ArabicVSRDataModule

    cfg = OmegaConf.create({
        "train_csv": str(splits_dir / "train.csv"),
        "val_csv": str(splits_dir / "val.csv"),
        "test_csv": str(splits_dir / "test.csv"),
        "roi_dir": str(roi_dir),
        "tokenizer_path": str(tiny_tokenizer._sp.serialized_model_proto()),  # placeholder
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "max_seq_len": 30,
        "persistent_workers": False,
        "augmentation": {
            "horizontal_flip_p": 0.0,
            "temporal_jitter_max": 0,
            "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        },
    })

    # Patch tokenizer loading to use the session-scoped tokenizer
    from unittest.mock import patch
    with patch("lsparabic.data.datamodule.ArabicSentencePieceTokenizer", return_value=tiny_tokenizer):
        dm = ArabicVSRDataModule(cfg)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

    assert "frames" in batch
    assert "labels" in batch
    assert batch["frames"].ndim == 5  # (B, T, C, H, W)
    assert batch["frames"].dtype == torch.float32
