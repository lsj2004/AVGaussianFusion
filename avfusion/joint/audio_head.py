from __future__ import annotations

import torch
from torch import Tensor, nn


class JointAudioHead(nn.Module):
    def __init__(self, num_points: int, top_k: int | None = None):
        super().__init__()
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        self.num_points = int(num_points)
        self.top_k = min(int(top_k), num_points) if top_k is not None else num_points
        self.audio_opacity = nn.Parameter(torch.zeros(num_points, 1))
        self.mono_gain = nn.Parameter(torch.zeros(num_points, 1))
        self.diff_gain = nn.Parameter(torch.zeros(num_points, 1))
        self.delay_offset = nn.Parameter(torch.zeros(num_points, 1))
        self.attenuation_logit = nn.Parameter(torch.zeros(num_points, 1))

    @property
    def active_count(self) -> int:
        return self.top_k

    def forward(self, state: dict[str, Tensor], source_audio: Tensor) -> Tensor:
        if source_audio.ndim != 2 or source_audio.shape[0] != 2:
            raise ValueError(f"source_audio must have shape (2, samples), got {tuple(source_audio.shape)}")
        xyz = state["xyz"]
        opacity = state["opacity"]
        scores = (opacity + self.audio_opacity).reshape(-1)
        indices = torch.argsort(scores, descending=True, stable=True)[: self.top_k]
        selected_opacity = scores.index_select(0, indices).reshape(-1, 1)
        weights = torch.softmax(selected_opacity, dim=0)
        selected_xyz = xyz.index_select(0, indices)
        distance = selected_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        attenuation = torch.sigmoid(self.attenuation_logit.index_select(0, indices)) / distance
        weights = weights * attenuation
        weights = weights / weights.sum().clamp_min(1e-6)
        mono = torch.tanh((weights * self.mono_gain.index_select(0, indices)).sum())
        diff = torch.tanh((weights * self.diff_gain.index_select(0, indices)).sum())
        mono_source = source_audio.to(mono).mean(dim=0, keepdim=True)
        left = mono_source * (1 + mono + diff)
        right = mono_source * (1 + mono - diff)
        return torch.cat([left, right], dim=0)
