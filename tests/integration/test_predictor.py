"""Integration test: VSRPredictor produces a string from a silent video."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def test_predictor_returns_string(tmp_path, dummy_video_path, tiny_tokenizer):
    from lsparabic.data.frame_extractor import VideoFrameExtractor
    from lsparabic.inference.beam_search import BeamSearchDecoder
    from lsparabic.inference.predictor import VSRPredictor
    from lsparabic.inference.sliding_window import SlidingWindowProcessor
    from lsparabic.models.conformer import ConformerEncoder
    from lsparabic.models.decoder import CTCDecoder
    from lsparabic.models.visual_encoder import ResNet3D
    from lsparabic.models.vsr_model import VSRModel

    vocab = tiny_tokenizer.vocab_size
    model = VSRModel(
        visual_encoder=ResNet3D(output_dim=64),
        sequence_model=ConformerEncoder(input_dim=64, d_model=64, num_layers=1),
        decoder=CTCDecoder(input_dim=64, vocab_size=vocab),
        proj_dim=64,
    )

    # Mock ROI cropper to avoid requiring MediaPipe in CI
    mock_cropper = MagicMock()
    mock_cropper.crop.return_value = np.zeros((20, 96, 96, 3), dtype=np.uint8)

    frame_extractor = VideoFrameExtractor(target_fps=5, max_frames=20)
    window_proc = SlidingWindowProcessor(window_size=10, stride=5)
    decoder = BeamSearchDecoder(tokenizer=tiny_tokenizer, beam_width=1)

    predictor = VSRPredictor(
        model=model,
        frame_extractor=frame_extractor,
        roi_cropper=mock_cropper,
        decoder=decoder,
        window_processor=window_proc,
        device="cpu",
    )

    result = predictor.predict(dummy_video_path)
    assert isinstance(result, str)
