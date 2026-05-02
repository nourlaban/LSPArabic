from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from loguru import logger


class ReferenceVoiceExtractor:
    """Extract a clean reference audio clip from a video for voice cloning.

    XTTS v2 needs 3–60 s of clean single-speaker speech at 22 050 Hz mono WAV.
    This class wraps ffmpeg to pull a segment from any video that contains audio.
    """

    def __init__(
        self,
        duration: float = 30.0,
        start_offset: float = 2.0,
        sample_rate: int = 22050,
        normalize_audio: bool = True,
    ) -> None:
        self.duration = duration
        self.start_offset = start_offset
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio

    def extract(self, video_path: Path, output_path: Path) -> Path:
        """Extract `duration` seconds of audio from `video_path` starting at
        `start_offset`, writing a 22 050 Hz mono 16-bit WAV to `output_path`.

        Returns `output_path`.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(self.start_offset),
            "-t", str(self.duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
        ]

        if self.normalize_audio:
            cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]

        cmd.append(str(output_path))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed extracting reference audio from {video_path}:\n{result.stderr}"
            )

        logger.info(
            f"Reference audio extracted: {output_path} "
            f"({self.duration}s @ {self.sample_rate}Hz from {video_path.name})"
        )
        return output_path

    def extract_best_segment(
        self,
        video_path: Path,
        output_path: Path,
        num_candidates: int = 3,
    ) -> Path:
        """Try multiple start offsets and pick the segment with highest RMS energy
        (loudest clear speech segment). Falls back to `extract()` on any error.
        """
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            return self.extract(video_path, output_path)

        video_duration = self._get_video_duration(video_path)
        max_start = max(0.0, video_duration - self.duration - self.start_offset)
        offsets = [self.start_offset + i * (max_start / max(num_candidates, 1))
                   for i in range(num_candidates)]

        best_path: Path | None = None
        best_rms = -1.0

        with tempfile.TemporaryDirectory() as tmp:
            for i, offset in enumerate(offsets):
                candidate = Path(tmp) / f"candidate_{i}.wav"
                try:
                    self._extract_segment(video_path, candidate, offset)
                    audio, _ = sf.read(str(candidate))
                    rms = float(np.sqrt(np.mean(audio ** 2)))
                    if rms > best_rms:
                        best_rms = rms
                        best_path = candidate
                except Exception:
                    continue

            if best_path and best_path.exists():
                import shutil
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(best_path, output_path)
                logger.info(f"Best reference segment (RMS={best_rms:.4f}) → {output_path}")
                return output_path

        return self.extract(video_path, output_path)

    def _extract_segment(self, video_path: Path, output_path: Path, start: float) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(start),
            "-t", str(self.duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg segment extraction failed")

    def _get_video_duration(self, video_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 60.0
