"""Synthesis script: lip-read a silent video and output it with cloned Arabic speech.

Full pipeline:
  1. Run the trained VSR model on the silent video → predicted Arabic text
  2. Extract a reference voice clip from a reference video of the same speaker
  3. Synthesize the predicted text in the speaker's voice via XTTS v2
  4. Mux the synthesized audio into the silent video → output .mp4

Usage:
  python scripts/synthesize.py \\
      synthesis.checkpoint_path=checkpoints/best.ckpt \\
      synthesis.silent_video=my_silent_clip.mp4 \\
      synthesis.reference_video=reference_with_audio.mp4 \\
      synthesis.output_video=output/result.mp4

Override reference audio directly (skip extraction):
  python scripts/synthesize.py ... synthesis.reference_audio=my_voice.wav

Burn Arabic subtitles into video:
  python scripts/synthesize.py ... synthesis.burn_subtitles=true
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from lsparabic.data.frame_extractor import VideoFrameExtractor
from lsparabic.data.roi_cropper import MediaPipeMouthCropper
from lsparabic.data.tokenizer import ArabicSentencePieceTokenizer
from lsparabic.inference.beam_search import BeamSearchDecoder
from lsparabic.inference.predictor import VSRPredictor
from lsparabic.inference.sliding_window import SlidingWindowProcessor
from lsparabic.synthesis.audio_video_muxer import AudioVideoMuxer
from lsparabic.synthesis.synthesis_pipeline import SynthesisPipeline
from lsparabic.synthesis.tts_synthesizer import XTTSSynthesizer
from lsparabic.synthesis.voice_extractor import ReferenceVoiceExtractor
from lsparabic.training.lightning_module import VSRLightningModule
from lsparabic.training.teacher import WhisperTeacher
from lsparabic.utils.checkpoint_utils import build_model_from_checkpoint
from lsparabic.utils.logging_utils import setup_logging


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    scfg = cfg.synthesis
    device = scfg.device if torch.cuda.is_available() else "cpu"

    # ── Load VSR model ────────────────────────────────────────────────────────
    tokenizer = ArabicSentencePieceTokenizer(Path(cfg.data.tokenizer_path))
    model = build_model_from_checkpoint(scfg.checkpoint_path, tokenizer.vocab_size)
    teacher = WhisperTeacher()
    module = VSRLightningModule.load_from_checkpoint(
        scfg.checkpoint_path,
        cfg=cfg,
        model=model,
        teacher=teacher,
        tokenizer=tokenizer,
        map_location=device,
    )

    # ── Build VSRPredictor ────────────────────────────────────────────────────
    icfg = cfg.inference
    frame_extractor = VideoFrameExtractor(target_fps=cfg.preprocessing.target_fps)
    roi_cropper = MediaPipeMouthCropper(
        output_size=tuple(cfg.preprocessing.mouth_roi.output_size),
        crop_factor=cfg.preprocessing.mouth_roi.crop_factor,
    )
    window_proc = SlidingWindowProcessor(
        window_size=icfg.sliding_window.window_size,
        stride=icfg.sliding_window.stride,
        overlap_strategy=icfg.sliding_window.overlap_strategy,
    )
    beam_decoder = BeamSearchDecoder(
        tokenizer=tokenizer,
        beam_width=icfg.beam_search.beam_width,
        alpha=icfg.beam_search.alpha,
    )
    predictor = VSRPredictor(
        model=module.model,
        frame_extractor=frame_extractor,
        roi_cropper=roi_cropper,
        decoder=beam_decoder,
        window_processor=window_proc,
        device=device,
    )

    # ── Build synthesis components ────────────────────────────────────────────
    vcfg = scfg.voice_extraction
    voice_extractor = ReferenceVoiceExtractor(
        duration=vcfg.duration,
        start_offset=vcfg.start_offset,
        sample_rate=vcfg.sample_rate,
        normalize_audio=vcfg.normalize_audio,
    )

    tcfg = scfg.tts
    synthesizer = XTTSSynthesizer(
        device=device,
        temperature=tcfg.temperature,
        length_penalty=tcfg.length_penalty,
        repetition_penalty=tcfg.repetition_penalty,
        top_k=tcfg.top_k,
        top_p=tcfg.top_p,
        speed=tcfg.speed,
    )

    mcfg = scfg.muxer
    muxer = AudioVideoMuxer(
        audio_bitrate=mcfg.audio_bitrate,
        audio_codec=mcfg.audio_codec,
        preserve_metadata=mcfg.preserve_metadata,
        pad_audio=mcfg.pad_audio,
    )

    pipeline = SynthesisPipeline(
        predictor=predictor,
        voice_extractor=voice_extractor,
        synthesizer=synthesizer,
        muxer=muxer,
        burn_subtitles=scfg.burn_subtitles,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    reference_audio = Path(scfg.reference_audio) if scfg.reference_audio else None
    reference_video = Path(scfg.reference_video) if scfg.reference_video else None

    if reference_audio is None and reference_video is None:
        raise ValueError(
            "Provide either synthesis.reference_video (a video with audio from the target speaker) "
            "or synthesis.reference_audio (a pre-extracted WAV file)."
        )

    result = pipeline.run(
        silent_video=Path(scfg.silent_video),
        reference_video=reference_video,
        output_video=Path(scfg.output_video),
        reference_audio=reference_audio,
    )

    print(f"\n{'=' * 65}")
    print(f"  Predicted text : {result.predicted_text}")
    print(f"  Reference audio: {result.reference_audio}")
    print(f"  Output video   : {result.output_video}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
