from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from lsparabic.data.interfaces import (
    BaseASRLabeler,
    BaseFrameExtractor,
    BaseROICropper,
    BaseTokenizer,
)
from lsparabic.data.dataset import VSRSample


class PreprocessingPipeline:
    def __init__(
        self,
        cfg: DictConfig,
        extractor: BaseFrameExtractor,
        cropper: BaseROICropper,
        labeler: BaseASRLabeler,
        tokenizer: BaseTokenizer,
    ) -> None:
        self.cfg = cfg
        self.extractor = extractor
        self.cropper = cropper
        self.labeler = labeler
        self.tokenizer = tokenizer

    def run(
        self,
        video_dir: Path,
        output_dir: Path,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        roi_dir = output_dir / "mouth_rois"
        label_dir = output_dir / "labels"
        splits_dir = output_dir.parent / "splits"
        roi_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted(
            p for p in video_dir.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        logger.info(f"Found {len(video_files)} videos in {video_dir}")

        samples: list[VSRSample] = []
        for video_path in tqdm(video_files, desc="Preprocessing"):
            sample = self._process_one(video_path, roi_dir, label_dir)
            if sample is not None:
                samples.append(sample)

        logger.info(f"Successfully processed {len(samples)} videos")
        self._write_splits(samples, splits_dir, val_ratio, test_ratio)

    def _process_one(
        self,
        video_path: Path,
        roi_dir: Path,
        label_dir: Path,
    ) -> VSRSample | None:
        video_id = video_path.stem
        roi_path = roi_dir / f"{video_id}.npy"
        label_path = label_dir / f"{video_id}.json"

        try:
            # Extract and crop mouth ROI
            if not roi_path.exists():
                frames = self.extractor.extract(video_path)
                roi = self.cropper.crop(frames)
                self._save_roi(roi, roi_path)
            else:
                roi = np.load(str(roi_path))

            # Transcribe audio
            if label_path.exists():
                label_text = json.loads(label_path.read_text())["text"]
            else:
                label_text = self.labeler.transcribe_from_video(video_path)
                label_path.write_text(json.dumps({"text": label_text}, ensure_ascii=False))

            token_ids = self.tokenizer.encode(label_text)
            return VSRSample(
                video_id=video_id,
                roi_path=roi_path,
                label_text=label_text,
                token_ids=token_ids,
            )

        except Exception as exc:
            logger.warning(f"Skipping {video_path.name}: {exc}")
            return None

    def _write_splits(
        self,
        samples: list[VSRSample],
        splits_dir: Path,
        val_ratio: float,
        test_ratio: float,
    ) -> None:
        def _bucket(video_id: str) -> str:
            h = int(hashlib.md5(video_id.encode()).hexdigest(), 16) % 100
            if h < int(test_ratio * 100):
                return "test"
            if h < int((test_ratio + val_ratio) * 100):
                return "val"
            return "train"

        rows: dict[str, list] = {"train": [], "val": [], "test": []}
        for s in samples:
            rows[_bucket(s.video_id)].append(
                {"video_id": s.video_id, "label_text": s.label_text}
            )

        for split, data in rows.items():
            path = splits_dir / f"{split}.csv"
            pd.DataFrame(data).to_csv(path, index=False)
            logger.info(f"Wrote {len(data)} rows to {path}")

    @staticmethod
    def _save_roi(roi: np.ndarray, path: Path) -> None:
        np.save(str(path), roi)
