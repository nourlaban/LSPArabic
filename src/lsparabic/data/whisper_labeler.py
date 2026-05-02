from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from lsparabic.data.interfaces import BaseASRLabeler


class WhisperASRLabeler(BaseASRLabeler):
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        language: str = "ar",
        initial_prompt: str | None = None,
        beam_size: int = 5,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.language = language
        self.initial_prompt = initial_prompt
        self.beam_size = beam_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            compute_type = "float16" if self.device == "cuda" else "int8"
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
            logger.info(f"Loaded Whisper {self.model_size} on {self.device}")
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        model = self._load_model()
        segments, _ = model.transcribe(
            str(audio_path),
            language=self.language,
            initial_prompt=self.initial_prompt,
            beam_size=self.beam_size,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        logger.debug(f"Transcribed {audio_path.name}: {text[:60]}...")
        return text.strip()

    def transcribe_from_video(self, video_path: Path, tmp_dir: Path | None = None) -> str:
        ctx = tempfile.TemporaryDirectory() if tmp_dir is None else None
        audio_dir = Path(ctx.name) if ctx else tmp_dir
        audio_path = audio_dir / f"{video_path.stem}.wav"
        try:
            self._extract_audio(video_path, audio_path)
            return self.transcribe(audio_path)
        finally:
            if ctx:
                ctx.cleanup()

    def _extract_audio(self, video_path: Path, out_path: Path) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn", "-ar", "16000", "-ac", "1",
            "-f", "wav", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {video_path}:\n{result.stderr}")
