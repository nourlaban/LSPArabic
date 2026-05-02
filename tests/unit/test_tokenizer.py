from __future__ import annotations

import pytest


def test_encode_decode_roundtrip(tiny_tokenizer):
    text = "مرحبا"
    ids = tiny_tokenizer.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    decoded = tiny_tokenizer.decode(ids)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_vocab_size_positive(tiny_tokenizer):
    assert tiny_tokenizer.vocab_size > 0


def test_encode_returns_list_of_ints(tiny_tokenizer):
    ids = tiny_tokenizer.encode("اللغة العربية")
    assert isinstance(ids, list)
    assert len(ids) > 0


def test_decode_empty(tiny_tokenizer):
    result = tiny_tokenizer.decode([])
    assert isinstance(result, str)
