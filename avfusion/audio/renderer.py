from __future__ import annotations

import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio.acoustic_gaussians import AcousticGaussianParameters


def render_audio(
    acoustic_carrier: AcousticCarrier,
    params: AcousticGaussianParameters,
    source_audio: torch.Tensor,
) -> torch.Tensor:
    if source_audio.ndim != 2 or source_audio.shape[0] != 2:
        raise ValueError(
            f"source_audio must have shape (2, samples), got {tuple(source_audio.shape)}"
        )
    device = params.mono_gain.device
    source_audio = source_audio.to(device=device, dtype=params.mono_gain.dtype)
    mono, diff = params.aggregate(acoustic_carrier.opacity)
    mono_source = source_audio.mean(dim=0, keepdim=True)
    left = mono_source * (1.0 + mono + diff)
    right = mono_source * (1.0 + mono - diff)
    return torch.cat([left, right], dim=0)
