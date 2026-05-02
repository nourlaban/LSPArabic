from __future__ import annotations

import pytest
import torch

mamba_ssm = pytest.importorskip("mamba_ssm", reason="mamba_ssm not installed; skipping Mamba tests")


def test_mamba_output_shape():
    from lsparabic.models.mamba_backend import MambaSSMEncoder
    model = MambaSSMEncoder(input_dim=64, d_model=64, num_layers=2)
    model.eval()
    x = torch.randn(2, 10, 64)
    lengths = torch.tensor([10, 8])
    with torch.no_grad():
        out = model(x, lengths)
    assert out.shape == (2, 10, 64)


def test_mamba_output_dim_property():
    from lsparabic.models.mamba_backend import MambaSSMEncoder
    model = MambaSSMEncoder(input_dim=32, d_model=64)
    assert model.output_dim == 64
