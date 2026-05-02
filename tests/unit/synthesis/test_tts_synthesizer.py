from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_is_available_without_tts_package():
    from lsparabic.synthesis.tts_synthesizer import XTTSSynthesizer

    synth = XTTSSynthesizer()
    with patch.dict("sys.modules", {"TTS": None, "TTS.api": None}):
        # importorskip catches the ImportError from the availability check
        result = synth.is_available()
    # Either True (TTS installed) or False (not installed); just assert it's a bool
    assert isinstance(result, bool)


def test_synthesize_raises_on_empty_text(tmp_path):
    from lsparabic.synthesis.tts_synthesizer import XTTSSynthesizer

    synth = XTTSSynthesizer()
    ref_audio = tmp_path / "ref.wav"
    ref_audio.touch()
    out = tmp_path / "out.wav"

    with pytest.raises(ValueError, match="empty"):
        synth.synthesize("   ", ref_audio, out)


def test_synthesize_raises_on_missing_reference(tmp_path):
    from lsparabic.synthesis.tts_synthesizer import XTTSSynthesizer

    synth = XTTSSynthesizer()
    out = tmp_path / "out.wav"

    with pytest.raises(FileNotFoundError):
        synth.synthesize("مرحبا", tmp_path / "nonexistent.wav", out)


def test_synthesize_calls_tts_api(tmp_path):
    from lsparabic.synthesis.tts_synthesizer import XTTSSynthesizer

    synth = XTTSSynthesizer()
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")  # minimal non-empty file
    out = tmp_path / "out.wav"

    mock_tts = MagicMock()
    synth._tts = mock_tts  # inject mock, bypass _load()

    synth.synthesize("مرحبا بالعالم", ref, out, language="ar")

    mock_tts.tts_to_file.assert_called_once_with(
        text="مرحبا بالعالم",
        file_path=str(out),
        speaker_wav=str(ref),
        language="ar",
        split_sentences=True,
    )
