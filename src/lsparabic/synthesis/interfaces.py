from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseVoiceSynthesizer(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path,
        language: str = "ar",
    ) -> Path:
        """Synthesize `text` in the voice of `reference_audio`.

        Returns the path to the generated WAV file.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the underlying model can be loaded on this machine."""
