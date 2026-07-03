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
    state = carrier.query(t)
    k = min(top_k, len(carrier))
    visual_indices = torch.argsort(state.opacity.reshape(-1), descending=True)[:k]

    return AcousticCarrier(
        xyz=state.xyz.index_select(0, visual_indices).detach(),
        opacity=state.opacity.index_select(0, visual_indices).detach(),
        visual_indices=visual_indices.detach().cpu(),
    )
