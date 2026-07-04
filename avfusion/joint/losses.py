from __future__ import annotations

from torch import Tensor, nn

from avfusion.joint.model import JointAVGaussianModel


def geometry_regularization(model: JointAVGaussianModel) -> Tensor:
    means = model.shared_gaussians.means
    opacities = model.shared_gaussians.opacities
    means_reg = nn.functional.mse_loss(means, model.means_init)
    opacity_reg = nn.functional.mse_loss(opacities.sigmoid(), model.opacities_init.sigmoid())
    return means_reg + opacity_reg


def joint_loss(
    model: JointAVGaussianModel,
    pred_rgb: Tensor,
    target_rgb: Tensor,
    pred_audio: Tensor,
    target_audio: Tensor,
    weights: dict[str, float],
) -> dict[str, Tensor]:
    zero = pred_rgb.sum() * 0
    rgb_l1 = weights.get("rgb_l1", 0.0) * nn.functional.l1_loss(pred_rgb, target_rgb.to(pred_rgb))
    audio_l1 = weights.get("audio_l1", 0.0) * nn.functional.l1_loss(pred_audio, target_audio.to(pred_audio))
    geo = weights.get("geo", 0.0) * geometry_regularization(model)
    total = zero + rgb_l1 + audio_l1 + geo
    return {"total": total, "rgb_l1": rgb_l1.detach(), "audio_l1": audio_l1.detach(), "geo": geo.detach()}
