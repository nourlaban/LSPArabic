"""Inference script: run VSRPredictor on a silent video and print Arabic text."""

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
from lsparabic.training.lightning_module import VSRLightningModule
from lsparabic.training.teacher import WhisperTeacher
from lsparabic.utils.checkpoint_utils import build_model_from_checkpoint
from lsparabic.utils.logging_utils import setup_logging


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    device = cfg.inference.device if torch.cuda.is_available() else "cpu"

    tokenizer = ArabicSentencePieceTokenizer(Path(cfg.data.tokenizer_path))

    model = build_model_from_checkpoint(cfg.inference.checkpoint_path, tokenizer.vocab_size)

    teacher = WhisperTeacher()
    module = VSRLightningModule.load_from_checkpoint(
        cfg.inference.checkpoint_path,
        cfg=cfg,
        model=model,
        teacher=teacher,
        tokenizer=tokenizer,
        map_location=device,
    )

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
        beta=icfg.beam_search.beta,
        lm_model_path=icfg.beam_search.lm_model_path,
    )
    predictor = VSRPredictor(
        model=module.model,
        frame_extractor=frame_extractor,
        roi_cropper=roi_cropper,
        decoder=beam_decoder,
        window_processor=window_proc,
        device=device,
    )

    result = predictor.predict(Path(icfg.video_path))
    print(f"\n{'=' * 60}")
    print(f"Predicted Arabic text:\n{result}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
