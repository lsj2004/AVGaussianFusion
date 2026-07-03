from __future__ import annotations

import torch
from torch import nn


class AcousticGaussianParameters(nn.Module):
    def __init__(self, num_points: int):
        super().__init__()
        self.mono_gain = nn.Parameter(torch.zeros(num_points, 1))
        self.diff_gain = nn.Parameter(torch.zeros(num_points, 1))

    def aggregate(self, opacity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        opacity = opacity.to(device=self.mono_gain.device, dtype=self.mono_gain.dtype)
        weights = opacity / opacity.sum().clamp_min(1e-6)
        mono = torch.tanh((weights * self.mono_gain).sum())
        diff = torch.tanh((weights * self.diff_gain).sum())
        return mono, diff
