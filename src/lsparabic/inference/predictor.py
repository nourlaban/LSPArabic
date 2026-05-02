from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from lsparabic.data.interfaces import BaseFrameExtractor, BaseROICropper
from lsparabic.inference.interfaces import BaseDecoder
from lsparabic.inference.sliding_window import SlidingWindowProcessor
from lsparabic.models.vsr_model import VSRModel
from lsparabic.utils.arabic_text import normalize_arabic


class VSRPredictor:
    """End-to-end inference: silent MP4 video → predicted Arabic text."""

    def __init__(
        self,
        model: VSRModel,
        frame_extractor: BaseFrameExtractor,
        roi_cropper: BaseROICropper,
        decoder: BaseDecoder,
        window_processor: SlidingWindowProcessor,
        device: str = "cuda",
        normalize_output: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.frame_extractor = frame_extractor
        self.roi_cropper = roi_cropper
        self.decoder = decoder
        self.window_processor = window_processor
        self.device = device
        self.normalize_output = normalize_output

        # Normalization constants (ImageNet stats used during training)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def predict(self, video_path: Path) -> str:
        frames = self.frame_extractor.extract(video_path)
        roi_seq = self.roi_cropper.crop(frames)            # (T, H, W, 3) uint8
        windows = self.window_processor.process(roi_seq)   # list of (W, H, W, 3)

        window_preds: list[str] = []
        for window in windows:
            logits, lengths = self._forward_window(window)
            texts = self.decoder.decode(logits, lengths)
            window_preds.append(texts[0])

        result = self.window_processor.merge_predictions(window_preds)
        if self.normalize_output:
            result = normalize_arabic(result)
        return result

    def predict_batch(self, video_paths: list[Path]) -> list[str]:
        return [self.predict(p) for p in video_paths]

    def _forward_window(self, window: np.ndarray) -> tuple[Tensor, Tensor]:
        """Convert a single (T, H, W, 3) window to model logits."""
        # (T, H, W, 3) -> (1, T, 3, H, W) float32
        t = window.shape[0]
        tensor = torch.from_numpy(window).float() / 255.0
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)  # (1, T, 3, H, W)
        tensor = (tensor - self._mean) / self._std
        lengths = torch.tensor([t], device=self.device)
        out = self.model(tensor, lengths)
        return out["logits"], out["lengths"]
