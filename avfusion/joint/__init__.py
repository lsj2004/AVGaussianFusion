"""Joint audio-visual Gaussian route."""

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSDependencyError, FTGSRendererBridge
from avfusion.joint.model import JointAVGaussianModel

__all__ = [
    "FTGSDependencyError",
    "FTGSRendererBridge",
    "JointAVGaussianModel",
    "JointAudioHead",
]
