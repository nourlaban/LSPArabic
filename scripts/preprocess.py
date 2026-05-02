"""Preprocessing script: extract mouth ROIs and Whisper transcripts from raw videos."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from lsparabic.data.frame_extractor import VideoFrameExtractor
from lsparabic.data.preprocessing_pipeline import PreprocessingPipeline
from lsparabic.data.roi_cropper import MediaPipeMouthCropper
from lsparabic.data.tokenizer import ArabicSentencePieceTokenizer, ArabicTokenizerTrainer
from lsparabic.data.whisper_labeler import WhisperASRLabeler
from lsparabic.utils.logging_utils import setup_logging


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    pcfg = cfg.preprocessing

    extractor = VideoFrameExtractor(
        target_fps=pcfg.target_fps,
        max_frames=pcfg.max_frames_per_video,
    )
    cropper = MediaPipeMouthCropper(
        output_size=tuple(pcfg.mouth_roi.output_size),
        crop_factor=pcfg.mouth_roi.crop_factor,
        temporal_smooth_alpha=pcfg.mouth_roi.temporal_smooth_alpha,
        static_image_mode=pcfg.face_mesh.static_image_mode,
        max_num_faces=pcfg.face_mesh.max_num_faces,
        min_detection_confidence=pcfg.face_mesh.min_detection_confidence,
        min_tracking_confidence=pcfg.face_mesh.min_tracking_confidence,
    )
    labeler = WhisperASRLabeler(
        model_size=pcfg.whisper.model_size,
        device=pcfg.whisper.device,
        language=pcfg.whisper.language,
        initial_prompt=pcfg.whisper.initial_prompt,
        beam_size=pcfg.whisper.beam_size,
    )

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    tokenizer_dir = Path("data/tokenizer")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Train tokenizer on all transcripts (collected after labeling)
    # We first run pipeline with a placeholder tokenizer, then retrain
    trainer = ArabicTokenizerTrainer(
        vocab_size=pcfg.tokenizer.vocab_size,
        model_type=pcfg.tokenizer.model_type,
        character_coverage=pcfg.tokenizer.character_coverage,
    )
    spm_prefix = tokenizer_dir / "arabic_spm"

    # First pass: generate labels only to build corpus
    import json
    import tempfile
    label_dir = processed_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    ) if raw_dir.exists() else []

    if video_files:
        for video_path in video_files:
            label_path = label_dir / f"{video_path.stem}.json"
            if not label_path.exists():
                text = labeler.transcribe_from_video(video_path)
                label_path.write_text(json.dumps({"text": text}, ensure_ascii=False))

        # Build corpus file and train tokenizer
        corpus_file = tokenizer_dir / "corpus.txt"
        with corpus_file.open("w", encoding="utf-8") as f:
            for lp in label_dir.glob("*.json"):
                data = json.loads(lp.read_text())
                f.write(data["text"] + "\n")
        trainer.train(corpus_file, spm_prefix)

    tokenizer = ArabicSentencePieceTokenizer(Path(str(spm_prefix) + ".model"))

    pipeline = PreprocessingPipeline(
        cfg=pcfg,
        extractor=extractor,
        cropper=cropper,
        labeler=labeler,
        tokenizer=tokenizer,
    )
    pipeline.run(raw_dir, processed_dir)


if __name__ == "__main__":
    main()
