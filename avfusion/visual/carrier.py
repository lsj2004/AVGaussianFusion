from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CarrierState:
    xyz: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacity: torch.Tensor


class FrozenVisualCarrier:
    def __init__(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        times: torch.Tensor,
        durations: torch.Tensor,
        velocities: torch.Tensor,
        max_duration: float,
    ) -> None:
        self.means = means.detach().float()
        self.scales = scales.detach().float()
        self.quats = quats.detach().float()
        self.opacities = opacities.detach().float()
        self.times = times.detach().float()
        self.durations = durations.detach().float()
        self.velocities = velocities.detach().float()
        self.max_duration = float(max_duration)
        self.assert_frozen()

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def query(self, t: float | torch.Tensor) -> CarrierState:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), dtype=self.means.dtype, device=self.means.device)
        t = t.reshape(1, 1).to(self.means)

        xyz = self.means + (t - self.times) * self.velocities
        temporal_opacity = torch.exp(-0.5 * ((t - self.times) / self._temporal_scale()) ** 2)
        opacity = torch.sigmoid(self.opacities) * temporal_opacity

        return CarrierState(
            xyz=xyz.detach(),
            scales=self.scales.detach(),
            quats=self.quats.detach(),
            opacity=opacity.detach(),
        )

    def assert_frozen(self) -> None:
        for name in ("means", "scales", "quats", "opacities", "times", "durations", "velocities"):
            tensor = getattr(self, name)
            if tensor.requires_grad:
                raise RuntimeError(f"carrier tensor {name} must be frozen")

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "means": self.means,
                "scales": self.scales,
                "quats": self.quats,
                "opacities": self.opacities,
                "times": self.times,
                "durations": self.durations,
                "velocities": self.velocities,
                "max_duration": self.max_duration,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FrozenVisualCarrier":
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**data)

    def _temporal_scale(self) -> torch.Tensor:
        if self.max_duration == float("inf"):
            return torch.exp(self.durations)
        return self.max_duration / 6.0 * torch.sigmoid(self.durations)
