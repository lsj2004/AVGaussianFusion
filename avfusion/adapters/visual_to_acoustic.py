from __future__ import annotations

from dataclasses import dataclass

import torch

from avfusion.visual.carrier import FrozenVisualCarrier


@dataclass(frozen=True)
class AcousticCarrier:
    xyz: torch.Tensor
    opacity: torch.Tensor
    visual_indices: torch.Tensor


def select_topk_acoustic_carrier(
    carrier: FrozenVisualCarrier, t: float, top_k: int
) -> AcousticCarrier:
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    state = carrier.query(t)
    k = min(top_k, len(carrier))
    visual_indices = torch.argsort(
        state.opacity.reshape(-1), descending=True, stable=True
    )[:k]

    return AcousticCarrier(
        xyz=state.xyz.index_select(0, visual_indices).detach().clone(),
        opacity=state.opacity.index_select(0, visual_indices).detach().clone(),
        visual_indices=visual_indices.detach().cpu(),
    )
