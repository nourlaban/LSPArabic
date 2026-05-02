from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from lsparabic.models.interfaces import BaseSequenceModel, BaseVisualEncoder


class VSRModel(nn.Module):
    """Assembled VSR model: VisualEncoder → SequenceModel → Decoder.

    Accepts any concrete implementations of the abstract interfaces,
    making it trivially swappable (Conformer ↔ Mamba, ResNet3D ↔ other).
    """

    def __init__(
        self,
        visual_encoder: BaseVisualEncoder,
        sequence_model: BaseSequenceModel,
        decoder: nn.Module,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.sequence_model = sequence_model
        self.decoder = decoder

        # Bridge projection when encoder and sequence model dims differ
        enc_dim = visual_encoder.output_dim
        seq_dim = sequence_model.output_dim
        self.bridge = nn.Linear(enc_dim, proj_dim) if enc_dim != proj_dim else nn.Identity()

    def forward(self, frames: Tensor, frame_lengths: Tensor) -> dict[str, Tensor]:
        """
        frames:        (B, T, C, H, W)
        frame_lengths: (B,)
        Returns dict with:
            logits:      (B, T', vocab+1)  CTC log-probs
            encoder_out: (B, T', D)         intermediate features
            lengths:     (B,)               valid sequence lengths after encoding
        """
        visual_feats = self.visual_encoder(frames)           # (B, T', enc_dim)
        visual_feats = self.bridge(visual_feats)             # (B, T', proj_dim)

        # Adjust lengths for temporal downsampling in ResNet3D
        t_out = visual_feats.size(1)
        t_in = frames.size(1)
        scale = t_out / max(t_in, 1)
        enc_lengths = (frame_lengths.float() * scale).long().clamp(min=1, max=t_out)

        encoder_out = self.sequence_model(visual_feats, enc_lengths)  # (B, T', D)
        logits = self.decoder(encoder_out)                             # (B, T', vocab+1)

        return {
            "logits": logits,
            "encoder_out": encoder_out,
            "lengths": enc_lengths,
        }

    def get_intermediate_features(self, frames: Tensor, frame_lengths: Tensor) -> Tensor:
        """Return Conformer/Mamba hidden states for FeatureRegressionLoss."""
        visual_feats = self.visual_encoder(frames)
        visual_feats = self.bridge(visual_feats)
        t_out = visual_feats.size(1)
        t_in = frames.size(1)
        scale = t_out / max(t_in, 1)
        enc_lengths = (frame_lengths.float() * scale).long().clamp(min=1, max=t_out)
        return self.sequence_model(visual_feats, enc_lengths)
