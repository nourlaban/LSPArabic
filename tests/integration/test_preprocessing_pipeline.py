"""Integration test: PreprocessingPipeline on a real synthetic video."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_cropper():
    cropper = MagicMock()
    cropper.crop.return_value = np.zeros((30, 96, 96, 3), dtype=np.uint8)
    return cropper


def _make_mock_labeler():
    labeler = MagicMock()
    labeler.transcribe_from_video.return_value = "مرحبا بالعالم"
    return labeler


def test_pipeline_creates_outputs(tmp_path, dummy_video_path, tiny_tokenizer):
    from lsparabic.data.frame_extractor import VideoFrameExtractor
    from lsparabic.data.preprocessing_pipeline import PreprocessingPipeline
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({})

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Copy dummy video to raw dir
    import shutil
    shutil.copy(dummy_video_path, raw_dir / "vid001.mp4")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    extractor = VideoFrameExtractor(target_fps=5, max_frames=30)
    cropper = _make_mock_cropper()
    labeler = _make_mock_labeler()

    pipeline = PreprocessingPipeline(
        cfg=cfg,
        extractor=extractor,
        cropper=cropper,
        labeler=labeler,
        tokenizer=tiny_tokenizer,
    )
    pipeline.run(raw_dir, processed_dir)

    roi_path = processed_dir / "mouth_rois" / "vid001.npy"
    assert roi_path.exists(), "ROI .npy file should be created"

    label_path = processed_dir / "labels" / "vid001.json"
    assert label_path.exists(), "Label .json file should be created"

    splits_dir = tmp_path / "splits"
    assert (splits_dir / "train.csv").exists() or (splits_dir / "test.csv").exists()
