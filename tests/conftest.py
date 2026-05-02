"""Shared pytest fixtures for LSPArabic tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Dummy video fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dummy_video_path(tmp_path_factory) -> Path:
    """2-second, 25fps, 320×240 synthetic RGB video."""
    tmp = tmp_path_factory.mktemp("videos")
    out = tmp / "dummy.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 25.0, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(50):  # 2 seconds at 25fps
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return out


# ---------------------------------------------------------------------------
# Tiny ROI array fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dummy_roi(tmp_path_factory) -> Path:
    """Save a small (30, 96, 96, 3) mouth ROI .npy file."""
    tmp = tmp_path_factory.mktemp("rois")
    path = tmp / "vid001.npy"
    rng = np.random.default_rng(1)
    roi = rng.integers(0, 255, (30, 96, 96, 3), dtype=np.uint8)
    np.save(str(path), roi)
    return path


# ---------------------------------------------------------------------------
# Tiny SentencePiece tokenizer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    """Train a minimal Arabic SPM model and return the tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError:
        pytest.skip("sentencepiece not installed")

    tmp = tmp_path_factory.mktemp("tokenizer")
    corpus = tmp / "corpus.txt"
    corpus.write_text(
        "\n".join([
            "مرحبا بكم",
            "أهلاً وسهلاً",
            "كيف حالك",
            "شكراً جزيلاً",
            "اللغة العربية",
            "الذكاء الاصطناعي",
            "قراءة الشفاه",
            "التعلم العميق",
            "الشبكات العصبية",
            "معالجة اللغة الطبيعية",
        ]),
        encoding="utf-8",
    )
    prefix = tmp / "spm"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=50,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

    from lsparabic.data.tokenizer import ArabicSentencePieceTokenizer
    return ArabicSentencePieceTokenizer(Path(str(prefix) + ".model"))


# ---------------------------------------------------------------------------
# Mock Whisper teacher
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_whisper_teacher(mocker):
    """WhisperTeacher that returns fixed-shape tensors without loading Whisper."""
    import torch
    teacher = MagicMock()
    teacher.forward.return_value = {
        "logits": torch.zeros(1, 10, 51865),
        "hidden_states": torch.zeros(1, 10, 1280),
    }
    return teacher


# ---------------------------------------------------------------------------
# Tiny split CSV
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_split_csv(tmp_path_factory, dummy_roi):
    """Create train/val/test CSVs pointing at the dummy ROI."""
    tmp = tmp_path_factory.mktemp("splits")
    roi_dir = dummy_roi.parent
    for split in ("train", "val", "test"):
        csv = tmp / f"{split}.csv"
        csv.write_text("video_id,label_text\nvid001,مرحبا\n", encoding="utf-8")
    return tmp, roi_dir
