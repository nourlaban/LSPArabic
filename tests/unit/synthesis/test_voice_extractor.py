from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_extract_calls_ffmpeg(tmp_path, dummy_video_path):
    from lsparabic.synthesis.voice_extractor import ReferenceVoiceExtractor

    extractor = ReferenceVoiceExtractor(duration=5.0, start_offset=0.0, normalize_audio=False)
    out = tmp_path / "ref.wav"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        extractor.extract(dummy_video_path, out)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "ffmpeg" in cmd
    assert str(dummy_video_path) in cmd
    assert str(out) in cmd
    assert "-t" in cmd
    assert "5.0" in cmd


def test_extract_raises_on_ffmpeg_failure(tmp_path, dummy_video_path):
    from lsparabic.synthesis.voice_extractor import ReferenceVoiceExtractor

    extractor = ReferenceVoiceExtractor(normalize_audio=False)
    out = tmp_path / "ref.wav"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            extractor.extract(dummy_video_path, out)


def test_output_directory_is_created(tmp_path, dummy_video_path):
    from lsparabic.synthesis.voice_extractor import ReferenceVoiceExtractor

    extractor = ReferenceVoiceExtractor(normalize_audio=False)
    nested = tmp_path / "deep" / "nested" / "ref.wav"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        extractor.extract(dummy_video_path, nested)

    assert nested.parent.exists()
