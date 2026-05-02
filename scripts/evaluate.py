"""Evaluation script: compute WER/CER on the test split."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lsparabic.data.datamodule import ArabicVSRDataModule
from lsparabic.data.tokenizer import ArabicSentencePieceTokenizer
from lsparabic.models.vsr_model import VSRModel
from lsparabic.training.lightning_module import VSRLightningModule
from lsparabic.training.teacher import WhisperTeacher
from lsparabic.utils.logging_utils import setup_logging


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()

    tokenizer = ArabicSentencePieceTokenizer(Path(cfg.data.tokenizer_path))
    OmegaConf.update(cfg, "model.decoder.vocab_size", tokenizer.vocab_size, merge=True)

    visual_encoder = instantiate(cfg.model.visual_encoder)
    sequence_model = instantiate(cfg.model.sequence_model)
    decoder = instantiate(cfg.model.decoder)
    model = VSRModel(
        visual_encoder=visual_encoder,
        sequence_model=sequence_model,
        decoder=decoder,
        proj_dim=cfg.model.proj_dim,
    )

    teacher = WhisperTeacher(
        model_size=cfg.training.teacher.model_size,
        device=cfg.training.teacher.device,
    )

    module = VSRLightningModule.load_from_checkpoint(
        cfg.inference.checkpoint_path,
        cfg=cfg,
        model=model,
        teacher=teacher,
        tokenizer=tokenizer,
    )

    datamodule = ArabicVSRDataModule(cfg.data)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False)
    results = trainer.test(module, datamodule=datamodule)
    print(results)


if __name__ == "__main__":
    main()
