from __future__ import annotations

from torch import Tensor, nn

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge


class JointAVGaussianModel(nn.Module):
    def __init__(self, bridge: FTGSRendererBridge, audio_head: JointAudioHead):
        super().__init__()
        self.bridge = bridge
        self.shared_gaussians = bridge.gaussians
        self.audio_head = audio_head
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
            parameter.requires_grad_(name in {"means", "opacities", "velocity_model"})
