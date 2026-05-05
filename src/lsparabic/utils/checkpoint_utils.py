from __future__ import annotations

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lsparabic.models.vsr_model import VSRModel


def build_model_from_checkpoint(ckpt_path: str | Path, vocab_size: int) -> VSRModel:
    """Instantiate VSRModel using the model config stored inside the checkpoint.

    This avoids requiring callers to re-specify model=mamba (or any other
    architecture override) on the command line — the architecture is recovered
    directly from the saved hyperparameters.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    model_cfg_dict = hparams.get("model_config")
    if model_cfg_dict is None:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has no 'model_config' in hyper_parameters. "
            "Re-train with the current code to embed the config in the checkpoint."
        )

    model_cfg: DictConfig = OmegaConf.create(model_cfg_dict)
    OmegaConf.update(model_cfg, "decoder.vocab_size", vocab_size, merge=True)

    visual_encoder = instantiate(model_cfg.visual_encoder)
    sequence_model = instantiate(model_cfg.sequence_model)
    decoder = instantiate(model_cfg.decoder)
    return VSRModel(
        visual_encoder=visual_encoder,
        sequence_model=sequence_model,
        decoder=decoder,
        proj_dim=model_cfg.proj_dim,
    )
