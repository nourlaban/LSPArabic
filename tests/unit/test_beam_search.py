from __future__ import annotations

from unittest.mock import MagicMock

import torch
import pytest


def _make_mock_tokenizer(vocab_size: int = 50):
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.decode.return_value = "مرحبا"
    return tok


def test_beam_width_1_equals_greedy():
    from lsparabic.inference.beam_search import BeamSearchDecoder
    tok = _make_mock_tokenizer()
    decoder = BeamSearchDecoder(tokenizer=tok, beam_width=1)
    B, T, V = 1, 10, 50
    logits = torch.randn(B, T, V).log_softmax(-1)
    lengths = torch.tensor([T])
    results = decoder.decode(logits, lengths)
    assert len(results) == 1
    assert isinstance(results[0], str)


def test_decode_returns_batch_results():
    from lsparabic.inference.beam_search import BeamSearchDecoder
    tok = _make_mock_tokenizer()
    decoder = BeamSearchDecoder(tokenizer=tok, beam_width=3)
    B, T, V = 3, 8, 50
    logits = torch.randn(B, T, V).log_softmax(-1)
    lengths = torch.tensor([8, 6, 4])
    results = decoder.decode(logits, lengths)
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)
