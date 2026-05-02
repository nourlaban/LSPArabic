from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from lsparabic.data.dataset import ArabicVSRDataset
from lsparabic.data.tokenizer import ArabicSentencePieceTokenizer
from lsparabic.data.transforms import VideoAugmentations


class ArabicVSRDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = ArabicSentencePieceTokenizer(Path(cfg.tokenizer_path))
        self._train_ds: ArabicVSRDataset | None = None
        self._val_ds: ArabicVSRDataset | None = None
        self._test_ds: ArabicVSRDataset | None = None

    def setup(self, stage: str = "") -> None:
        roi_dir = Path(self.cfg.roi_dir)
        aug = VideoAugmentations(self.cfg.augmentation)

        if stage in ("fit", ""):
            self._train_ds = ArabicVSRDataset(
                split_csv=Path(self.cfg.train_csv),
                roi_dir=roi_dir,
                tokenizer=self.tokenizer,
                transform=aug,
                max_seq_len=self.cfg.max_seq_len,
            )
            self._val_ds = ArabicVSRDataset(
                split_csv=Path(self.cfg.val_csv),
                roi_dir=roi_dir,
                tokenizer=self.tokenizer,
                transform=None,
                max_seq_len=self.cfg.max_seq_len,
            )

        if stage in ("test", ""):
            self._test_ds = ArabicVSRDataset(
                split_csv=Path(self.cfg.test_csv),
                roi_dir=roi_dir,
                tokenizer=self.tokenizer,
                transform=None,
                max_seq_len=self.cfg.max_seq_len,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.get("persistent_workers", False),
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: list[dict]) -> dict[str, Tensor | list]:
        frames = torch.stack([b["frames"] for b in batch])
        frame_lengths = torch.stack([b["frame_lengths"] for b in batch])
        label_lengths = torch.stack([b["label_lengths"] for b in batch])

        # Pad labels to max length in batch
        max_label_len = int(label_lengths.max().item())
        labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
        for i, b in enumerate(batch):
            llen = b["label_lengths"].item()
            labels[i, :llen] = b["labels"][:llen]

        return {
            "frames": frames,
            "frame_lengths": frame_lengths,
            "labels": labels,
            "label_lengths": label_lengths,
            "video_ids": [b["video_id"] for b in batch],
        }
