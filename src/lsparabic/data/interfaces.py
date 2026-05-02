from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseFrameExtractor(ABC):
    @abstractmethod
    def extract(self, video_path: Path) -> list[np.ndarray]:
        """Return list of BGR frames sampled at target_fps."""

    @abstractmethod
    def get_fps(self, video_path: Path) -> float:
        """Return the native frame rate of the video."""


class BaseROICropper(ABC):
    @abstractmethod
    def crop(self, frames: list[np.ndarray]) -> np.ndarray:
        """Return mouth ROI sequence as (T, H, W, C) uint8 array."""


class BaseASRLabeler(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Return raw Arabic transcript for the given audio file."""


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode Arabic text to token id list."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode token id list to Arabic text."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
