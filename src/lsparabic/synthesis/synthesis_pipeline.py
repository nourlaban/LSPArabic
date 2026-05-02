from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from lsparabic.synthesis.audio_video_muxer import AudioVideoMuxer
from lsparabic.synthesis.interfaces import BaseVoiceSynthesizer
from lsparabic.synthesis.voice_extractor import ReferenceVoiceExtractor


@dataclass
class SynthesisResult:
    predicted_text: str
    reference_audio: Path
    synthesized_audio: Path
    output_video: Path


class SynthesisPipeline:
    """End-to-end pipeline: silent video → lip reading → voice clone → output video.

    Flow:
        1. Extract mouth ROIs from silent video
        2. Predict Arabic text via VSRPredictor
        3. Extract reference voice clip from a reference video (with audio)
        4. Synthesize Arabic text in the reference speaker's voice via XTTS v2
        5. Mux synthesized audio into the silent video → final output
    """

    def __init__(
        self,
        predictor,                         # VSRPredictor
        voice_extractor: ReferenceVoiceExtractor,
        synthesizer: BaseVoiceSynthesizer,
        muxer: AudioVideoMuxer,
        work_dir: Path | None = None,
        burn_subtitles: bool = False,
    ) -> None:
        self.predictor = predictor
        self.voice_extractor = voice_extractor
        self.synthesizer = synthesizer
        self.muxer = muxer
        self.work_dir = work_dir
        self.burn_subtitles = burn_subtitles

    def run(
        self,
        silent_video: Path,
        reference_video: Path,
        output_video: Path,
        reference_audio: Path | None = None,
    ) -> SynthesisResult:
        """Process one silent video end-to-end.

        Args:
            silent_video:    The input video without audio.
            reference_video: A video of the same speaker with clean audio.
                             Used to extract the reference voice clip.
            output_video:    Destination path for the final video with speech.
            reference_audio: If provided, skip voice extraction and use this
                             WAV directly (must be 22 050 Hz mono, 3–60 s).

        Returns a SynthesisResult with all intermediate paths and the text.
        """
        ctx = tempfile.TemporaryDirectory() if self.work_dir is None else None
        tmp = Path(ctx.name) if ctx else self.work_dir
        tmp.mkdir(parents=True, exist_ok=True)

        try:
            # ── Step 1: Predict text from silent video ─────────────────────
            logger.info(f"[1/4] Predicting text from: {silent_video.name}")
            predicted_text = self.predictor.predict(silent_video)
            logger.info(f"      Predicted: {predicted_text!r}")

            if not predicted_text.strip():
                raise ValueError("VSR model returned empty text — check the input video.")

            # ── Step 2: Extract reference voice clip ───────────────────────
            if reference_audio is None:
                ref_audio_path = tmp / "reference_voice.wav"
                logger.info(f"[2/4] Extracting reference voice from: {reference_video.name}")
                reference_audio = self.voice_extractor.extract_best_segment(
                    reference_video, ref_audio_path
                )
            else:
                logger.info(f"[2/4] Using provided reference audio: {reference_audio.name}")

            # ── Step 3: Synthesize speech ──────────────────────────────────
            synth_path = tmp / "synthesized_speech.wav"
            logger.info("[3/4] Synthesizing Arabic speech with voice cloning …")
            synth_path = self.synthesizer.synthesize_long(
                text=predicted_text,
                reference_audio=reference_audio,
                output_path=synth_path,
                language="ar",
            )

            # ── Step 4: Mux audio into video ───────────────────────────────
            logger.info(f"[4/4] Muxing audio into video → {output_video.name}")
            if self.burn_subtitles:
                self.muxer.mux_with_subtitle(
                    video_path=silent_video,
                    audio_path=synth_path,
                    subtitle_text=predicted_text,
                    output_path=output_video,
                )
            else:
                self.muxer.mux(
                    video_path=silent_video,
                    audio_path=synth_path,
                    output_path=output_video,
                )

            logger.info(f"Done. Output: {output_video}")
            return SynthesisResult(
                predicted_text=predicted_text,
                reference_audio=reference_audio,
                synthesized_audio=synth_path,
                output_video=output_video,
            )

        finally:
            if ctx:
                ctx.cleanup()

    def run_batch(
        self,
        silent_videos: list[Path],
        reference_video: Path,
        output_dir: Path,
        reference_audio: Path | None = None,
    ) -> list[SynthesisResult]:
        """Process multiple silent videos using the same reference speaker."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract reference audio once for the whole batch
        if reference_audio is None:
            ref_audio = output_dir / "reference_voice.wav"
            logger.info(f"Extracting shared reference voice from: {reference_video.name}")
            reference_audio = self.voice_extractor.extract_best_segment(
                reference_video, ref_audio
            )

        results = []
        for video in silent_videos:
            out = output_dir / f"{video.stem}_with_speech{video.suffix}"
            try:
                result = self.run(
                    silent_video=video,
                    reference_video=reference_video,
                    output_video=out,
                    reference_audio=reference_audio,
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"Failed on {video.name}: {exc}")
        return results
