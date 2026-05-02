from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lsparabic.models.interfaces import BaseVisualEncoder


class ResNet3DBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=(1, stride, stride), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D(BaseVisualEncoder):
    """3D ResNet-18 variant for visual speech recognition.

    Input:  (B, T, C, H, W)
    Output: (B, T, output_dim)
    """

    def __init__(self, output_dim: int = 512, pretrained: bool = False) -> None:
        super().__init__()
        self._output_dim = output_dim

        # Initial 3D convolution: large temporal kernel for motion capture
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))

        if output_dim != 512:
            self.proj = nn.Linear(512, output_dim)
        else:
            self.proj = nn.Identity()

        self._init_weights()

    def _make_layer(
        self, in_ch: int, out_ch: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_ch),
            )
        layers = [ResNet3DBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResNet3DBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C, H, W) -> rearrange to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)            # (B, 512, T', 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 512, T')
        x = x.permute(0, 2, 1)        # (B, T', 512)
        x = self.proj(x)               # (B, T', output_dim)
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim
