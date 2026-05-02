"""Integration test: SynthesisPipeline wires all synthesis components end-to-end."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_predictor(text: str = "مرحبا بالعالم"):
    predictor = MagicMock()
    predictor.predict.return_value = text
    return predictor


def _make_mock_extractor(tmp_path: Path):
    extractor = MagicMock()
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_bytes(b"RIFF")
    extractor.extract_best_segment.return_value = ref_wav
    return extractor


def _make_mock_synthesizer(tmp_path: Path):
    synthesizer = MagicMock()
    synth_wav = tmp_path / "speech.wav"
    synth_wav.write_bytes(b"RIFF")
    synthesizer.synthesize_long.return_value = synth_wav
    return synthesizer


def _make_mock_muxer(tmp_path: Path):
    muxer = MagicMock()
    # muxer.mux returns the output path
    muxer.mux.side_effect = lambda video_path, audio_path, output_path, **_: (
        output_path.__class__(output_path).parent.mkdir(parents=True, exist_ok=True)
        or _touch_and_return(output_path)
    )
    return muxer


def _touch_and_return(path: Path) -> Path:
    path.touch()
    return path


def test_pipeline_run_calls_all_stages(tmp_path, dummy_video_path):
    from lsparabic.synthesis.synthesis_pipeline import SynthesisPipeline

    predictor = _make_mock_predictor()
    extractor = _make_mock_extractor(tmp_path)
    synthesizer = _make_mock_synthesizer(tmp_path)
    muxer = _make_mock_muxer(tmp_path)

    pipeline = SynthesisPipeline(
        predictor=predictor,
        voice_extractor=extractor,
        synthesizer=synthesizer,
        muxer=muxer,
        work_dir=tmp_path / "work",
    )

    out_video = tmp_path / "output.mp4"
    result = pipeline.run(
        silent_video=dummy_video_path,
        reference_video=dummy_video_path,
        output_video=out_video,
    )

    assert result.predicted_text == "مرحبا بالعالم"
    predictor.predict.assert_called_once_with(dummy_video_path)
    extractor.extract_best_segment.assert_called_once()
    synthesizer.synthesize_long.assert_called_once()
    muxer.mux.assert_called_once()


def test_pipeline_raises_on_empty_prediction(tmp_path, dummy_video_path):
    from lsparabic.synthesis.synthesis_pipeline import SynthesisPipeline

    predictor = _make_mock_predictor(text="   ")
    pipeline = SynthesisPipeline(
        predictor=predictor,
        voice_extractor=_make_mock_extractor(tmp_path),
        synthesizer=_make_mock_synthesizer(tmp_path),
        muxer=_make_mock_muxer(tmp_path),
        work_dir=tmp_path / "work",
    )

    with pytest.raises(ValueError, match="empty text"):
        pipeline.run(
            silent_video=dummy_video_path,
            reference_video=dummy_video_path,
            output_video=tmp_path / "out.mp4",
        )


def test_pipeline_uses_provided_reference_audio(tmp_path, dummy_video_path):
    from lsparabic.synthesis.synthesis_pipeline import SynthesisPipeline

    predictor = _make_mock_predictor()
    extractor = _make_mock_extractor(tmp_path)
    synthesizer = _make_mock_synthesizer(tmp_path)
    muxer = _make_mock_muxer(tmp_path)

    pipeline = SynthesisPipeline(
        predictor=predictor,
        voice_extractor=extractor,
        synthesizer=synthesizer,
        muxer=muxer,
        work_dir=tmp_path / "work",
    )

    provided_ref = tmp_path / "my_voice.wav"
    provided_ref.write_bytes(b"RIFF")

    pipeline.run(
        silent_video=dummy_video_path,
        reference_video=dummy_video_path,
        output_video=tmp_path / "out.mp4",
        reference_audio=provided_ref,
    )

    # voice extractor should NOT be called when reference_audio is provided
    extractor.extract_best_segment.assert_not_called()
