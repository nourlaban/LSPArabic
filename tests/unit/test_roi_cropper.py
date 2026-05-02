from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lsparabic.data.roi_cropper import MediaPipeMouthCropper, MOUTH_LANDMARK_INDICES


def _make_frames(n: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(n)]


def _make_landmark(x_rel: float = 0.5, y_rel: float = 0.7):
    lm = MagicMock()
    lm.x = x_rel
    lm.y = y_rel
    return lm


def test_output_shape_with_mock_mediapipe():
    """Cropper returns (T, 96, 96, 3) when landmarks are detected."""
    n_frames = 5
    frames = _make_frames(n_frames)

    # Build a mock FaceMesh that always returns landmarks
    mock_results = MagicMock()
    mock_face = MagicMock()
    mock_face.landmark = [_make_landmark(0.5, 0.7) for _ in range(478)]
    mock_results.multi_face_landmarks = [mock_face]

    cropper = MediaPipeMouthCropper(output_size=(96, 96), crop_factor=1.5)

    with patch("mediapipe.solutions.face_mesh.FaceMesh") as mock_fm_cls:
        mock_fm = MagicMock()
        mock_fm.process.return_value = mock_results
        mock_fm_cls.return_value = mock_fm

        roi = cropper.crop(frames)

    assert roi.shape == (n_frames, 96, 96, 3)
    assert roi.dtype == np.uint8


def test_fallback_when_no_face_detected():
    """Returns (T, 96, 96, 3) even when MediaPipe detects nothing (fallback crop)."""
    frames = _make_frames(3)
    mock_results = MagicMock()
    mock_results.multi_face_landmarks = None

    cropper = MediaPipeMouthCropper(output_size=(96, 96))

    with patch("mediapipe.solutions.face_mesh.FaceMesh") as mock_fm_cls:
        mock_fm = MagicMock()
        mock_fm.process.return_value = mock_results
        mock_fm_cls.return_value = mock_fm

        roi = cropper.crop(frames)

    assert roi.shape == (3, 96, 96, 3)


def test_landmark_indices_unique():
    assert len(MOUTH_LANDMARK_INDICES) == len(set(MOUTH_LANDMARK_INDICES))
