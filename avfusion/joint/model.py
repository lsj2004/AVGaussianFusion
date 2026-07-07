from __future__ import annotations

from torch import Tensor, nn

from avfusion.joint.audio_head import AudioGSMaskedSpectralHead, JointAudioHead, SpectralJointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge


class JointAVGaussianModel(nn.Module):
    def __init__(
        self,
        bridge: FTGSRendererBridge,
        audio_head: JointAudioHead | SpectralJointAudioHead | AudioGSMaskedSpectralHead,
    ):
        super().__init__()
        self.bridge = bridge
        self.shared_gaussians = bridge.gaussians
        self.audio_head = audio_head
        self._shared_requires_grad_init = {
            name: parameter.requires_grad for name, parameter in self.shared_gaussians.named_parameters()
        }
        self.register_buffer("means_init", self.shared_gaussians.means.detach().clone())
        self.register_buffer("opacities_init", self.shared_gaussians.opacities.detach().clone())

    def render_rgb(self, batch: dict, sh_degree: int | None = None) -> Tensor:
        return self.bridge.render_rgb(batch, sh_degree=sh_degree)

    def render_audio(self, t: Tensor, source_audio: Tensor) -> Tensor:
        return self.audio_head(self.bridge.query_state(t), source_audio)

    def parameter_groups(self, shared_lr: float, audio_lr: float) -> list[dict]:
        return [
            {"name": "shared", "params": list(self.shared_gaussians.parameters()), "lr": shared_lr},
            {"name": "audio", "params": list(self.audio_head.parameters()), "lr": audio_lr},
        ]

    def freeze_shared(self) -> None:
        for parameter in self.shared_gaussians.parameters():
            parameter.requires_grad_(False)

    def unfreeze_shared_geometry(self) -> None:
        for name, parameter in self.shared_gaussians.named_parameters():
            is_geometry_parameter = name in {"means", "opacities", "velocity_model"} or name.startswith("velocity_model.")
            parameter.requires_grad_(is_geometry_parameter and self._shared_requires_grad_init.get(name, False))
