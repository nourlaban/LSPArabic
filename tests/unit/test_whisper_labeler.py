from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lsparabic.data.whisper_labeler import WhisperASRLabeler


def test_transcribe_returns_string(tmp_path):
    dummy_audio = tmp_path / "audio.wav"
    dummy_audio.touch()

    segment = MagicMock()
    segment.text = " مرحبا بالعالم "

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([segment], MagicMock())

    labeler = WhisperASRLabeler(model_size="tiny", device="cpu")
    labeler._model = mock_model

    result = labeler.transcribe(dummy_audio)
    assert isinstance(result, str)
    assert "مرحبا" in result


def test_extract_audio_calls_ffmpeg(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    audio_out = tmp_path / "audio.wav"

    labeler = WhisperASRLabeler()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        labeler._extract_audio(video, audio_out)

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "ffmpeg" in args
    assert str(video) in args
    assert str(audio_out) in args
