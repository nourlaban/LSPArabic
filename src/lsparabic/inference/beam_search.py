from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from lsparabic.data.interfaces import BaseTokenizer
from lsparabic.inference.interfaces import BaseDecoder


@dataclass
class BeamHypothesis:
    tokens: list[int] = field(default_factory=list)
    score: float = 0.0
    last_token: int = 0  # tracks last non-blank for CTC collapse


class BeamSearchDecoder(BaseDecoder):
    """CTC beam search decoder with optional length normalization."""

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        beam_width: int = 10,
        alpha: float = 1.0,
        beta: float = 0.0,
        blank_idx: int = 0,
        lm_model_path: Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.blank_idx = blank_idx
        self._lm = None

        if lm_model_path is not None:
            self._load_lm(lm_model_path)

    def _load_lm(self, path: Path) -> None:
        try:
            from pyctcdecode import build_ctc_decoder
            vocab = [self.tokenizer.id_to_piece(i) for i in range(self.tokenizer.vocab_size)]
            self._lm_decoder = build_ctc_decoder(vocab, kenlm_model=str(path), alpha=self.alpha, beta=self.beta)
        except ImportError:
            pass

    def decode(self, logits: Tensor, lengths: Tensor) -> list[str]:
        """logits: (B, T, V) log-probs; returns list[str] of decoded Arabic text."""
        if self._lm is not None:
            return self._decode_with_lm(logits, lengths)
        return self._decode_ctc_beam(logits, lengths)

    def _decode_ctc_beam(self, logits: Tensor, lengths: Tensor) -> list[str]:
        results = []
        probs = logits.exp().cpu()  # (B, T, V)

        for b, tlen in enumerate(lengths.tolist()):
            seq_probs = probs[b, :tlen]  # (T, V)
            beams: list[BeamHypothesis] = [BeamHypothesis()]

            for t in range(seq_probs.size(0)):
                frame = seq_probs[t]  # (V,)
                new_beams: list[BeamHypothesis] = []

                for beam in beams:
                    for token_id in range(frame.size(0)):
                        p = float(frame[token_id])
                        if p < 1e-10:
                            continue
                        if token_id == self.blank_idx:
                            # Blank: keep hypothesis unchanged, update score
                            new_beams.append(BeamHypothesis(
                                tokens=list(beam.tokens),
                                score=beam.score + torch.log(frame[token_id] + 1e-10).item(),
                                last_token=beam.last_token,
                            ))
                        elif token_id == beam.last_token:
                            # Same token: only extend if blank separator was seen
                            # (simplified: just accumulate blank path)
                            new_beams.append(BeamHypothesis(
                                tokens=list(beam.tokens),
                                score=beam.score + torch.log(frame[token_id] + 1e-10).item(),
                                last_token=token_id,
                            ))
                        else:
                            new_beams.append(BeamHypothesis(
                                tokens=beam.tokens + [token_id],
                                score=beam.score + torch.log(frame[token_id] + 1e-10).item(),
                                last_token=token_id,
                            ))

                # Merge duplicates and keep top-k
                beams = self._merge_and_prune(new_beams)

            best = max(beams, key=lambda h: h.score / max(len(h.tokens) ** self.alpha, 1))
            results.append(self.tokenizer.decode(best.tokens))

        return results

    def _merge_and_prune(self, beams: list[BeamHypothesis]) -> list[BeamHypothesis]:
        """Merge beams with identical token sequences, keep top beam_width."""
        merged: dict[tuple, BeamHypothesis] = {}
        for b in beams:
            key = tuple(b.tokens)
            if key not in merged or b.score > merged[key].score:
                merged[key] = b
        sorted_beams = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        return sorted_beams[: self.beam_width]

    def _decode_with_lm(self, logits: Tensor, lengths: Tensor) -> list[str]:
        results = []
        log_probs_np = logits.cpu().numpy()
        for b, tlen in enumerate(lengths.tolist()):
            seq = log_probs_np[b, :tlen]
            results.append(self._lm_decoder.decode(seq))
        return results
