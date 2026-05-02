from __future__ import annotations

import pytest
import numpy as np

from lsparabic.data.frame_extractor import VideoFrameExtractor, FrameExtractionError


def test_extract_returns_frames(dummy_video_path):
    extractor = VideoFrameExtractor(target_fps=25, max_frames=50)
    frames = extractor.extract(dummy_video_path)
    assert len(frames) > 0
    assert len(frames) <= 50
    assert frames[0].ndim == 3  # (H, W, C)
    assert frames[0].shape[2] == 3


def test_get_fps(dummy_video_path):
    extractor = VideoFrameExtractor()
    fps = extractor.get_fps(dummy_video_path)
    assert fps > 0


def test_extract_invalid_path():
    extractor = VideoFrameExtractor()
    with pytest.raises(FrameExtractionError):
        extractor.extract("/nonexistent/path/video.mp4")


def test_frame_sampling_reduces_count(dummy_video_path):
    """Sampling at lower FPS should produce fewer frames."""
    hi = VideoFrameExtractor(target_fps=25, max_frames=500)
    lo = VideoFrameExtractor(target_fps=10, max_frames=500)
    frames_hi = hi.extract(dummy_video_path)
    frames_lo = lo.extract(dummy_video_path)
    assert len(frames_lo) <= len(frames_hi)
