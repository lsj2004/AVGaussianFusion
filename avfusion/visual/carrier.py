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
        marginal_gates: torch.Tensor | None = None,
    ) -> None:
        self.means = means.detach().clone().float()
        self.scales = scales.detach().clone().float()
        self.quats = quats.detach().clone().float()
        self.opacities = opacities.detach().clone().float()
        self.times = times.detach().clone().float()
        self.durations = durations.detach().clone().float()
        self.velocities = velocities.detach().clone().float()
        if marginal_gates is None:
            marginal_gates = torch.full((len(self.means), 1), -1.0)
        self.marginal_gates = marginal_gates.detach().clone().float()
        self.max_duration = float(max_duration)
        self.assert_frozen()

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def query(self, t: float | torch.Tensor) -> CarrierState:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), dtype=self.means.dtype, device=self.means.device)
        t = t.reshape(1, 1).to(self.means)

        xyz = self.means + (t - self.times) * self.velocities
        gate = torch.sigmoid(20 * self.marginal_gates)
        gaussian_opacity = torch.exp(
            -0.5 * ((t - self.times) / self._temporal_scale()) ** 2
        )
        temporal_opacity = gate + (1 - gate) * gaussian_opacity
        opacity = torch.sigmoid(self.opacities) * temporal_opacity

        return CarrierState(
            xyz=xyz.detach(),
            scales=self.scales.detach().clone(),
            quats=self.quats.detach().clone(),
            opacity=opacity.detach(),
        )

    def assert_frozen(self) -> None:
        for name in (
            "means",
            "scales",
            "quats",
            "opacities",
            "times",
            "durations",
            "velocities",
            "marginal_gates",
        ):
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
                "marginal_gates": self.marginal_gates,
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
