from __future__ import annotations

import torch
import pytest

from lsparabic.models.visual_encoder import ResNet3D


def test_output_shape():
    model = ResNet3D(output_dim=512)
    model.eval()
    x = torch.randn(2, 10, 3, 96, 96)  # (B, T, C, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[2] == 512


def test_output_dim_property():
    model = ResNet3D(output_dim=256)
    assert model.output_dim == 256


def test_custom_output_dim():
    model = ResNet3D(output_dim=128)
    model.eval()
    x = torch.randn(1, 8, 3, 96, 96)
    with torch.no_grad():
        out = model(x)
    assert out.shape[-1] == 128
