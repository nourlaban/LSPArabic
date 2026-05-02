from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from torch import Tensor


class VideoAugmentations:
    """Augmentation pipeline applied per-sample (T, C, H, W) float32 in [0,1]."""

    def __init__(self, cfg: DictConfig) -> None:
        self.flip_p = cfg.get("horizontal_flip_p", 0.5)
        self.jitter_max = cfg.get("temporal_jitter_max", 2)
        cj = cfg.get("color_jitter", {})
        self.brightness = cj.get("brightness", 0.0)
        self.contrast = cj.get("contrast", 0.0)
        norm = cfg.get("normalize", {})
        self.mean = list(norm.get("mean", [0.485, 0.456, 0.406]))
        self.std = list(norm.get("std", [0.229, 0.224, 0.225]))

    def __call__(self, frames: Tensor) -> Tensor:
        """frames: (T, C, H, W) float32 in [0, 1]"""
        frames = self._random_horizontal_flip(frames)
        frames = self._random_color_jitter(frames)
        frames = self._temporal_jitter(frames)
        frames = self._normalize(frames)
        return frames

    def _random_horizontal_flip(self, frames: Tensor) -> Tensor:
        if random.random() < self.flip_p:
            return torch.flip(frames, dims=[-1])
        return frames

    def _random_color_jitter(self, frames: Tensor) -> Tensor:
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            frames = torch.clamp(frames * factor, 0.0, 1.0)
        if self.contrast > 0:
            mean = frames.mean(dim=[-2, -1], keepdim=True)
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            frames = torch.clamp(mean + factor * (frames - mean), 0.0, 1.0)
        return frames

    def _temporal_jitter(self, frames: Tensor) -> Tensor:
        if self.jitter_max <= 0:
            return frames
        t = frames.shape[0]
        jitter = random.randint(-self.jitter_max, self.jitter_max)
        if jitter == 0 or t <= 1:
            return frames
        if jitter > 0:
            # drop last |jitter| frames
            return frames[: max(1, t - jitter)]
        # drop first |jitter| frames
        return frames[min(t - 1, -jitter):]

    def _normalize(self, frames: Tensor) -> Tensor:
        mean = torch.tensor(self.mean, dtype=frames.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.std, dtype=frames.dtype).view(1, 3, 1, 1)
        return (frames - mean) / std
