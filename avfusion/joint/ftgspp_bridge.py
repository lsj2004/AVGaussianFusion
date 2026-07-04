from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class FTGSDependencyError(RuntimeError):
    pass


class FTGSRendererBridge:
    def __init__(self, gaussians: torch.nn.Module):
        self.gaussians = gaussians

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str | Path) -> "FTGSRendererBridge":
        try:
            importlib.import_module("ftgspp.models.gaussians")
            importlib.import_module("gsplat")
        except ImportError as error:
            raise FTGSDependencyError(
                "Route B joint training requires the FTGS++ environment with ftgspp and gsplat importable."
            ) from error
        gaussians = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(gaussians)

    def to(self, device: torch.device | str) -> "FTGSRendererBridge":
        self.gaussians.to(device)
        return self

    def render_rgb(self, batch: dict[str, Any], sh_degree: int | None = None) -> Tensor:
        time = torch.as_tensor(batch["time"])
        first_time = time.reshape(-1)[0]
        if time.numel() > 1 and not torch.allclose(time, first_time.expand_as(time)):
            raise ValueError(
                "FTGSRendererBridge.render_rgb requires all cameras in a batch "
                "to share the same time."
            )
        render_time = time if time.ndim == 0 else time[0]
        image, _, _ = self.gaussians(
            t=render_time,
            w2c=batch["w2c"],
            intrinsic=batch["intrinsic"],
            shape=(int(batch["height"]), int(batch["width"])),
            sh_degree=sh_degree,
        )
        return image

    def query_state(self, t: Tensor) -> dict[str, Tensor]:
        return {
            "xyz": self.gaussians.means_t(t),
            "opacity": self.gaussians.opacities_t(t),
            "velocity": self._velocity_at(t),
        }

    def _velocity_at(self, t: Tensor) -> Tensor:
        velocity_model = getattr(self.gaussians, "velocity_model")
        if isinstance(velocity_model, torch.Tensor):
            return velocity_model
        return self.gaussians.velocities_t(t)
