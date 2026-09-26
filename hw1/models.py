"""Network used for Homework 1.

The architecture follows ``home_work_one.md`` literally.  BatchNorm is not
listed in the table, so it is deliberately not added here.  Convolutions have
no bias, every convolution is followed by an in-place ReLU, and the model is
intended for FP32 inference in ``eval`` mode.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Sequential CNN for 3-channel square inputs and 100 output classes."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model() -> SmallCNN:
    """Construct the assignment model in FP32 evaluation mode."""

    return SmallCNN().float().eval()


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(2, 3, 224, 224)
    print(model)
    print("output:", tuple(model(x).shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
