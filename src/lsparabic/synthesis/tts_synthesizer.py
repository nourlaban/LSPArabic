from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from lsparabic.synthesis.interfaces import BaseVoiceSynthesizer


def _patch_torchaudio_if_needed() -> None:
    """Replace torchaudio.load with a soundfile backend when torchcodec fails to load.

    torchcodec has a strict PyTorch ABI requirement; when there is a version
    mismatch the shared library fails to dlopen and torchaudio.load raises
    RuntimeError at import time.  soundfile handles plain WAV/FLAC/OGG just
    fine for the reference-audio use case.
    """
    try:
        import torchcodec.decoders  # noqa: F401
        return  # torchcodec loads cleanly — nothing to patch
    except Exception:
        pass

    import soundfile as sf
    import torchaudio

    def _sf_load(
        filepath,
        frame_offset: int = 0,
        num_frames: int = -1,
        _normalize: bool = True,
        channels_first: bool = True,
        **__,
    ):
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        if frame_offset:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        wav = torch.from_numpy(data.T if channels_first else data)
        return wav, sr

    torchaudio.load = _sf_load
    logger.info("torchaudio.load → soundfile fallback (torchcodec ABI mismatch)")


class XTTSSynthesizer(BaseVoiceSynthesizer):
    """Arabic voice cloning TTS using Coqui XTTS v2.

    Zero-shot voice cloning: clones the speaker from a 3–60 s reference clip
    and synthesises any Arabic text in that voice without fine-tuning.

    Install: pip install TTS
    Model:   tts_models/multilingual/multi-dataset/xtts_v2 (auto-downloaded ~1.7 GB)
    """

    MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
        self,
        device: str | None = None,
        temperature: float = 0.65,
        length_penalty: float = 1.0,
        repetition_penalty: float = 2.0,
        top_k: int = 50,
        top_p: float = 0.85,
        speed: float = 1.0,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.speed = speed
        self._tts = None

    def _load(self) -> None:
        if self._tts is not None:
            return
        _patch_torchaudio_if_needed()
        try:
            from TTS.api import TTS as CoquiTTS
        except ImportError as e:
            raise ImportError(
                "TTS package is required for XTTSSynthesizer. "
                "Install with: pip install 'lsparabic[synthesis]'"
            ) from e

        logger.info(f"Loading XTTS v2 on {self.device} …")
        self._tts = CoquiTTS(self.MODEL_ID).to(self.device)
        logger.info("XTTS v2 loaded.")

    def is_available(self) -> bool:
        try:
            from TTS.api import TTS  # noqa: F401
            return True
        except ImportError:
            return False

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path,
        language: str = "ar",
    ) -> Path:
        """Synthesize `text` in the voice from `reference_audio`.

        Args:
            text:            Arabic text to speak.
            reference_audio: Path to a 3–60 s WAV with the target speaker.
            output_path:     Destination WAV file path.
            language:        ISO 639-1 language code (default "ar" for Arabic).

        Returns the path to the written WAV file.
        """
        self._load()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        if not text.strip():
            raise ValueError("Text to synthesize must not be empty.")

        logger.info(f"Synthesizing [{len(text)} chars] → {output_path.name}")
        self._tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=str(reference_audio),
            language=language,
            split_sentences=True,
        )
        logger.info(f"Synthesis complete: {output_path}")
        return output_path

    def synthesize_long(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path,
        language: str = "ar",
        chunk_size: int = 200,
    ) -> Path:
        """Split long texts into chunks and concatenate the audio segments.

        XTTS v2 degrades on very long inputs (>~300 characters). This method
        splits on sentence boundaries and concatenates the resulting clips.
        """
        import re
        import tempfile

        import numpy as np
        import soundfile as sf

        # Split on Arabic sentence terminators: ، . ! ؟
        sentences = re.split(r"(?<=[.!?؟،])\s+", text.strip())
        if not sentences:
            return self.synthesize(text, reference_audio, output_path, language)

        # Group sentences into chunks that fit within chunk_size chars
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= chunk_size:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)

        if len(chunks) == 1:
            return self.synthesize(chunks[0], reference_audio, output_path, language)

        with tempfile.TemporaryDirectory() as tmp:
            parts: list[np.ndarray] = []
            sr: int = 22050
            for i, chunk in enumerate(chunks):
                part_path = Path(tmp) / f"part_{i:04d}.wav"
                self.synthesize(chunk, reference_audio, part_path, language)
                audio, sr = sf.read(str(part_path))
                # Small silence gap between chunks (0.25 s)
                parts.append(audio)
                parts.append(np.zeros(int(sr * 0.25), dtype=audio.dtype))

            combined = np.concatenate(parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), combined, sr)

        logger.info(f"Long synthesis complete ({len(chunks)} chunks): {output_path}")
        return output_path
