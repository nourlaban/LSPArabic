"""Training script: fit the VSR model using PyTorch Lightning + Hydra config."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from lsparabic.data.datamodule import ArabicVSRDataModule
from lsparabic.data.tokenizer import ArabicSentencePieceTokenizer
from lsparabic.models.conformer import ConformerEncoder
from lsparabic.models.decoder import CTCDecoder
from lsparabic.models.visual_encoder import ResNet3D
from lsparabic.models.vsr_model import VSRModel
from lsparabic.training.lightning_module import VSRLightningModule
from lsparabic.training.teacher import WhisperTeacher
from lsparabic.utils.logging_utils import setup_logging


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    pl.seed_everything(cfg.seed)

    # Load tokenizer to resolve vocab_size
    tokenizer = ArabicSentencePieceTokenizer(Path(cfg.data.tokenizer_path))
    OmegaConf.update(cfg, "model.decoder.vocab_size", tokenizer.vocab_size, merge=True)

    # Instantiate model components via Hydra
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
        cache_dir=cfg.training.teacher.cache_dir,
    )

    lightning_module = VSRLightningModule(
        cfg=cfg,
        model=model,
        teacher=teacher,
        tokenizer=tokenizer,
    )

    datamodule = ArabicVSRDataModule(cfg.data)

    # Build callbacks
    tcfg = cfg.training.callbacks
    callbacks = [
        ModelCheckpoint(**OmegaConf.to_container(tcfg.model_checkpoint, resolve=True)),
        EarlyStopping(**OmegaConf.to_container(tcfg.early_stopping, resolve=True)),
        LearningRateMonitor(logging_interval=tcfg.lr_monitor.logging_interval),
    ]

    logger = instantiate(cfg.logger)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(lightning_module, datamodule=datamodule)
    trainer.test(lightning_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
