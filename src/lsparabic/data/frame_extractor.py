from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from lsparabic.data.interfaces import BaseFrameExtractor


class FrameExtractionError(RuntimeError):
    pass


class VideoFrameExtractor(BaseFrameExtractor):
    def __init__(self, target_fps: int = 25, max_frames: int = 500) -> None:
        self.target_fps = target_fps
        self.max_frames = max_frames

    def get_fps(self, video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def extract(self, video_path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Cannot open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps <= 0 or total_frames <= 0:
            cap.release()
            raise FrameExtractionError(f"Invalid video metadata: {video_path}")

        sample_indices = set(self._sample_frames(total_frames, source_fps))
        frames: list[np.ndarray] = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in sample_indices:
                frames.append(frame)
                if len(frames) >= self.max_frames:
                    break
            idx += 1

        cap.release()

        if not frames:
            raise FrameExtractionError(f"No frames extracted from: {video_path}")

        logger.debug(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames

    def _sample_frames(self, total_frames: int, source_fps: float) -> list[int]:
        """Compute frame indices to achieve target_fps from source_fps."""
        if source_fps <= self.target_fps:
            return list(range(min(total_frames, self.max_frames)))

        step = source_fps / self.target_fps
        indices = []
        pos = 0.0
        while pos < total_frames and len(indices) < self.max_frames:
            indices.append(int(pos))
            pos += step
        return indices
