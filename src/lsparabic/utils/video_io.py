from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Context-manager wrapper around cv2.VideoCapture."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame


class VideoWriter:
    """Context-manager wrapper around cv2.VideoWriter."""

    def __init__(
        self,
        path: Path,
        fps: float = 25.0,
        frame_size: tuple[int, int] = (640, 480),
        codec: str = "mp4v",
    ) -> None:
        self.path = path
        self.fps = fps
        self.frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self._writer.release()

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
