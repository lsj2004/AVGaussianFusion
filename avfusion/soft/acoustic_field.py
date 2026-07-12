from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class AcousticGaussianField(nn.Module):
    def __init__(
        self,
        means: Tensor,
        opacities: Tensor,
        velocity_model: Tensor,
        anchor_indices: Tensor,
        anchor_weights: Tensor,
        anchor_mask: Tensor,
        residual_mask: Tensor,
        times: Tensor | None = None,
        audio_opacity: Tensor | None = None,
        mono_response: Tensor | None = None,
        diff_response: Tensor | None = None,
        side_response: Tensor | None = None,
        distance_decay: Tensor | None = None,
        phase_delay: Tensor | None = None,
    ):
        super().__init__()
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(f"means must have shape (N, 3), got {tuple(means.shape)}")
        num_points = int(means.shape[0])
        if opacities.shape != (num_points, 1):
            raise ValueError(f"opacities must have shape ({num_points}, 1), got {tuple(opacities.shape)}")
        if velocity_model.shape != (num_points, 3):
            raise ValueError(
                f"velocity_model must have shape ({num_points}, 3), got {tuple(velocity_model.shape)}"
            )
        if times is None:
            times = torch.zeros(num_points, 1, dtype=means.dtype, device=means.device)
        if times.shape != (num_points, 1):
            raise ValueError(f"times must have shape ({num_points}, 1), got {tuple(times.shape)}")
        if anchor_indices.ndim != 2 or anchor_indices.shape[0] != num_points:
            raise ValueError("anchor_indices must have shape (N, K)")
        if anchor_weights.shape != anchor_indices.shape:
            raise ValueError("anchor_weights must match anchor_indices")
        if anchor_mask.shape != (num_points,):
            raise ValueError(f"anchor_mask must have shape ({num_points},), got {tuple(anchor_mask.shape)}")
        if residual_mask.shape != (num_points,):
            raise ValueError(f"residual_mask must have shape ({num_points},), got {tuple(residual_mask.shape)}")
        if mono_response is None:
            mono_response = torch.zeros(num_points, 257, dtype=means.dtype, device=means.device)
        if diff_response is None:
            diff_response = torch.zeros_like(mono_response)
        if side_response is None:
            side_response = torch.zeros_like(mono_response)
        if distance_decay is None:
            distance_decay = torch.zeros_like(mono_response)
        if phase_delay is None:
            phase_delay = torch.zeros_like(mono_response)
        if audio_opacity is None:
            audio_opacity = torch.zeros(num_points, 1, dtype=means.dtype, device=means.device)
        if mono_response.ndim != 2 or mono_response.shape[0] != num_points:
            raise ValueError("mono_response must have shape (N, F)")
        if diff_response.shape != mono_response.shape:
            raise ValueError("diff_response must match mono_response")
        if side_response.shape != mono_response.shape:
            raise ValueError("side_response must match mono_response")
        if distance_decay.shape != mono_response.shape:
            raise ValueError("distance_decay must match mono_response")
        if phase_delay.shape != mono_response.shape:
            raise ValueError("phase_delay must match mono_response")
        if audio_opacity.shape != (num_points, 1):
            raise ValueError(f"audio_opacity must have shape ({num_points}, 1), got {tuple(audio_opacity.shape)}")

        self.means = nn.Parameter(means.float())
        self.opacities = nn.Parameter(opacities.float())
        self.velocity_model = nn.Parameter(velocity_model.float())
        self.times = nn.Parameter(times.float())
        self.audio_opacity = nn.Parameter(audio_opacity.float())
        self.mono_response = nn.Parameter(mono_response.float())
        self.diff_response = nn.Parameter(diff_response.float())
        self.side_response = nn.Parameter(side_response.float())
        self.distance_decay = nn.Parameter(distance_decay.float())
        self.phase_delay = nn.Parameter(phase_delay.float())
        self.register_buffer("anchor_indices", anchor_indices.long(), persistent=True)
        self.register_buffer("anchor_weights", anchor_weights.float(), persistent=True)
        self.register_buffer("anchor_mask", anchor_mask.bool(), persistent=True)
        self.register_buffer("residual_mask", residual_mask.bool(), persistent=True)

    @property
    def num_points(self) -> int:
        return int(self.means.shape[0])

    @property
    def num_frequency_bins(self) -> int:
        return int(self.mono_response.shape[1])

    @classmethod
    def from_visual_state(
        cls,
        visual_state: dict[str, Tensor],
        num_points: int,
        anchored_fraction: float = 0.6,
        dynamic_fraction: float = 0.2,
        residual_jitter: float = 0.05,
        num_frequency_bins: int = 257,
    ) -> "AcousticGaussianField":
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if num_frequency_bins <= 0:
            raise ValueError(f"num_frequency_bins must be positive, got {num_frequency_bins}")
        if not 0 <= anchored_fraction <= 1 or not 0 <= dynamic_fraction <= 1:
            raise ValueError("anchored_fraction and dynamic_fraction must be in [0, 1]")
        if anchored_fraction + dynamic_fraction > 1:
            raise ValueError("anchored_fraction + dynamic_fraction must be <= 1")

        xyz = visual_state["xyz"].detach().float()
        opacity = visual_state["opacity"].detach().float().reshape(-1, 1)
        velocity = visual_state["velocity"].detach().float()
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("visual_state['xyz'] must have shape (M, 3)")
        if velocity.shape != xyz.shape:
            raise ValueError("visual_state['velocity'] must match xyz shape")
        if opacity.shape[0] != xyz.shape[0]:
            raise ValueError("visual_state['opacity'] must match xyz point count")

        num_visual = int(xyz.shape[0])
        anchored_count = int(round(num_points * anchored_fraction))
        dynamic_count = int(round(num_points * dynamic_fraction))
        anchored_count = min(anchored_count, num_points)
        dynamic_count = min(dynamic_count, num_points - anchored_count)
        coupled_count = anchored_count + dynamic_count

        motion_score = velocity.norm(dim=-1, keepdim=True)
        visual_scores = opacity.reshape(-1) + 0.1 * motion_score.reshape(-1)
        ranked = torch.argsort(visual_scores, descending=True, stable=True)
        repeated = ranked.repeat((num_points + num_visual - 1) // num_visual)[:num_points]
        means = xyz.index_select(0, repeated).clone()
        if residual_jitter > 0:
            means = means + residual_jitter * torch.randn_like(means)
        opacities = torch.zeros(num_points, 1, dtype=torch.float32)
        audio_opacity = torch.zeros(num_points, 1, dtype=torch.float32)
        velocities = velocity.index_select(0, repeated).clone()
        mono_response = torch.zeros(num_points, num_frequency_bins, dtype=torch.float32)
        freq = torch.linspace(-0.5, 0.5, num_frequency_bins, dtype=torch.float32).reshape(1, num_frequency_bins)
        diff_response = 1e-3 * freq.repeat(num_points, 1)
        side_response = torch.zeros(num_points, num_frequency_bins, dtype=torch.float32)
        distance_decay = torch.zeros(num_points, num_frequency_bins, dtype=torch.float32)
        phase_delay = torch.zeros(num_points, num_frequency_bins, dtype=torch.float32)

        anchor_indices = repeated.reshape(num_points, 1).clone()
        anchor_weights = torch.ones(num_points, 1, dtype=torch.float32)
        anchor_mask = torch.zeros(num_points, dtype=torch.bool)
        anchor_mask[:coupled_count] = True
        residual_mask = ~anchor_mask
        if residual_mask.any():
            velocities[residual_mask] = 0.0
        return cls(
            means=means,
            opacities=opacities,
            velocity_model=velocities,
            anchor_indices=anchor_indices,
            anchor_weights=anchor_weights,
            anchor_mask=anchor_mask,
            residual_mask=residual_mask,
            audio_opacity=audio_opacity,
            mono_response=mono_response,
            diff_response=diff_response,
            side_response=side_response,
            distance_decay=distance_decay,
            phase_delay=phase_delay,
        )

    def query(self, t: float | Tensor) -> dict[str, Tensor]:
        if not isinstance(t, Tensor):
            t = torch.tensor(float(t), dtype=self.means.dtype, device=self.means.device)
        t = t.reshape(1, 1).to(device=self.means.device, dtype=self.means.dtype)
        xyz = self.means + (t - self.times) * self.velocity_model
        return {
            "xyz": xyz,
            "opacity": self.opacities.sigmoid(),
            "velocity": self.velocity_model,
            "audio_opacity": self.audio_opacity,
            "mono_response": self.mono_response,
            "diff_response": self.diff_response,
            "side_response": self.side_response,
            "distance_decay": self.distance_decay,
            "phase_delay": self.phase_delay,
        }

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_points": self.num_points,
                "num_frequency_bins": self.num_frequency_bins,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AcousticGaussianField":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        kwargs = {
            "means": state["means"],
            "opacities": state["opacities"],
            "velocity_model": state["velocity_model"],
            "times": state["times"],
            "anchor_indices": state["anchor_indices"],
            "anchor_weights": state["anchor_weights"],
            "anchor_mask": state["anchor_mask"],
            "residual_mask": state["residual_mask"],
        }
        for key in ("audio_opacity", "mono_response", "diff_response", "side_response", "distance_decay", "phase_delay"):
            if key in state:
                kwargs[key] = state[key]
        field = cls(**kwargs)
        field.load_state_dict(state, strict=False)
        return field
