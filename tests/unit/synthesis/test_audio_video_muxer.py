from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_mux_calls_ffmpeg(tmp_path):
    from lsparabic.synthesis.audio_video_muxer import AudioVideoMuxer

    muxer = AudioVideoMuxer()
    video = tmp_path / "silent.mp4"
    audio = tmp_path / "speech.wav"
    out = tmp_path / "result.mp4"
    video.touch()
    audio.touch()

    with patch("subprocess.run") as mock_run, \
         patch.object(AudioVideoMuxer, "_get_duration", return_value=5.0):
        mock_run.return_value = MagicMock(returncode=0)
        muxer.mux(video, audio, out)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "ffmpeg" in cmd
    assert str(video) in cmd
    assert str(audio) in cmd
    assert str(out) in cmd
    assert "-c:v" in cmd
    assert "copy" in cmd


def test_mux_raises_on_ffmpeg_failure(tmp_path):
    from lsparabic.synthesis.audio_video_muxer import AudioVideoMuxer

    muxer = AudioVideoMuxer()
    video = tmp_path / "silent.mp4"
    audio = tmp_path / "speech.wav"
    video.touch()
    audio.touch()

    with patch("subprocess.run") as mock_run, \
         patch.object(AudioVideoMuxer, "_get_duration", return_value=5.0):
        mock_run.return_value = MagicMock(returncode=1, stderr="error msg")
        with pytest.raises(RuntimeError, match="ffmpeg mux failed"):
            muxer.mux(video, audio, tmp_path / "out.mp4")


def test_mux_with_audio_offset(tmp_path):
    from lsparabic.synthesis.audio_video_muxer import AudioVideoMuxer

    muxer = AudioVideoMuxer()
    video = tmp_path / "v.mp4"
    audio = tmp_path / "a.wav"
    video.touch()
    audio.touch()

    with patch("subprocess.run") as mock_run, \
         patch.object(AudioVideoMuxer, "_get_duration", return_value=10.0):
        mock_run.return_value = MagicMock(returncode=0)
        muxer.mux(video, audio, tmp_path / "out.mp4", audio_offset=1.5)

    cmd = mock_run.call_args[0][0]
    assert "-itsoffset" in cmd
    assert "1.5" in cmd
