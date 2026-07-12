from __future__ import annotations

import torch
from torch import Tensor

from avfusion.soft.acoustic_field import AcousticGaussianField


def _anchor_average(values: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
    gathered = values.index_select(0, indices.reshape(-1)).reshape(*indices.shape, values.shape[-1])
    weights = weights.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    return (gathered * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)


def soft_coupling_losses(
    field: AcousticGaussianField,
    acoustic_state: dict[str, Tensor],
    visual_state: dict[str, Tensor],
) -> dict[str, Tensor]:
    device = acoustic_state["xyz"].device
    anchor_mask = field.anchor_mask.to(device=device)
    if not anchor_mask.any():
        zero = acoustic_state["xyz"].sum() * 0
        return {"anchor": zero, "motion": zero, "activity": zero, "sparse": acoustic_state["opacity"].mean()}

    indices = field.anchor_indices.to(device=device)
    weights = field.anchor_weights.to(device=device)
    visual_xyz = visual_state["xyz"].detach().to(acoustic_state["xyz"])
    visual_velocity = visual_state["velocity"].detach().to(acoustic_state["velocity"])
    visual_opacity = visual_state["opacity"].detach().to(acoustic_state["opacity"])

    anchor_xyz = _anchor_average(visual_xyz, indices, weights)
    anchor_velocity = _anchor_average(visual_velocity, indices, weights)
    anchor_activity = _anchor_average(visual_opacity, indices, weights)

    anchor = torch.nn.functional.l1_loss(acoustic_state["xyz"][anchor_mask], anchor_xyz[anchor_mask])
    motion = torch.nn.functional.l1_loss(
        acoustic_state["velocity"][anchor_mask],
        anchor_velocity[anchor_mask],
    )
    activity = torch.nn.functional.mse_loss(
        acoustic_state["opacity"][anchor_mask],
        anchor_activity[anchor_mask],
    )
    sparse = acoustic_state["opacity"].mean()
    return {"anchor": anchor, "motion": motion, "activity": activity, "sparse": sparse}
