from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lsparabic.data.interfaces import BaseTokenizer


@dataclass
class VSRSample:
    video_id: str
    roi_path: Path
    label_text: str
    token_ids: list[int]


class ArabicVSRDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        roi_dir: Path,
        tokenizer: BaseTokenizer,
        transform: Callable | None = None,
        max_seq_len: int = 200,
    ) -> None:
        self.roi_dir = roi_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len

        df = pd.read_csv(split_csv)
        self.samples: list[VSRSample] = []
        for _, row in df.iterrows():
            ids = tokenizer.encode(str(row["label_text"]))
            self.samples.append(
                VSRSample(
                    video_id=str(row["video_id"]),
                    roi_path=roi_dir / f"{row['video_id']}.npy",
                    label_text=str(row["label_text"]),
                    token_ids=ids,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.samples[idx]
        roi = self._load_roi(sample.roi_path)  # (T, H, W, 3) uint8
        roi = self._pad_or_trim(roi, self.max_seq_len)  # (max_seq_len, H, W, 3)

        # (T, H, W, 3) -> (T, 3, H, W) float32 in [0,1]
        frames = torch.from_numpy(roi).permute(0, 3, 1, 2).float() / 255.0
        frame_len = torch.tensor(min(len(roi), self.max_seq_len), dtype=torch.long)

        if self.transform is not None:
            frames = self.transform(frames)

        labels = torch.tensor(sample.token_ids, dtype=torch.long)
        label_len = torch.tensor(len(sample.token_ids), dtype=torch.long)

        return {
            "frames": frames,
            "frame_lengths": frame_len,
            "labels": labels,
            "label_lengths": label_len,
            "video_id": sample.video_id,
        }

    @lru_cache(maxsize=512)
    def _load_roi(self, path: Path) -> np.ndarray:
        return np.load(str(path))

    def _pad_or_trim(self, roi: np.ndarray, max_len: int) -> np.ndarray:
        t = roi.shape[0]
        if t >= max_len:
            return roi[:max_len]
        pad = np.zeros((max_len - t, *roi.shape[1:]), dtype=roi.dtype)
        return np.concatenate([roi, pad], axis=0)
