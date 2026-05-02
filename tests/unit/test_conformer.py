from __future__ import annotations

import torch
import pytest

from lsparabic.models.conformer import ConformerEncoder


def test_output_shape():
    model = ConformerEncoder(input_dim=512, d_model=256, num_layers=2)
    model.eval()
    x = torch.randn(2, 15, 512)
    lengths = torch.tensor([15, 10])
    with torch.no_grad():
        out = model(x, lengths)
    assert out.shape == (2, 15, 256)


def test_output_dim_property():
    model = ConformerEncoder(input_dim=64, d_model=128)
    assert model.output_dim == 128


def test_masked_positions_dont_crash():
    """Ensure padding masks with short sequences do not cause NaN."""
    model = ConformerEncoder(input_dim=64, d_model=64, num_layers=1)
    model.eval()
    x = torch.randn(3, 20, 64)
    lengths = torch.tensor([20, 5, 1])
    with torch.no_grad():
        out = model(x, lengths)
    assert not torch.isnan(out).any()
