from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CTCDecoder(nn.Module):
    """Linear projection to vocab_size + blank, outputs log-softmax."""

    def __init__(self, input_dim: int, vocab_size: int) -> None:
        super().__init__()
        # +1 for CTC blank token (index 0)
        self.proj = nn.Linear(input_dim, vocab_size + 1)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D) -> (B, T, vocab_size+1) log-probs"""
        return F.log_softmax(self.proj(x), dim=-1)


class AttentionDecoder(nn.Module):
    """Autoregressive cross-attention decoder for seq2seq lip reading."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_len: int = 200,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_out: Tensor,
        tgt_tokens: Tensor,
        encoder_lengths: Tensor,
    ) -> Tensor:
        """
        encoder_out:     (B, T_src, D)
        tgt_tokens:      (B, T_tgt)
        encoder_lengths: (B,)
        Returns:         (B, T_tgt, vocab_size) log-probs
        """
        b, t = tgt_tokens.shape
        positions = torch.arange(t, device=tgt_tokens.device).unsqueeze(0)
        tgt = self.dropout(self.embed(tgt_tokens) + self.pos_embed(positions))

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(t, device=tgt.device)
        memory_key_padding_mask = self._lengths_to_mask(encoder_lengths, encoder_out.size(1))

        out = self.transformer(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return F.log_softmax(self.out_proj(out), dim=-1)

    @staticmethod
    def _lengths_to_mask(lengths: Tensor, max_len: int) -> Tensor:
        return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
