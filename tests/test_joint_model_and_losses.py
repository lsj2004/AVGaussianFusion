import torch

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.losses import geometry_regularization, joint_loss
from avfusion.joint.model import JointAVGaussianModel
from tests.test_joint_ftgspp_bridge import FakeGaussians


def test_joint_model_exposes_separate_parameter_groups():
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4))

    groups = model.parameter_groups(shared_lr=1e-5, audio_lr=1e-3)

    assert [group["name"] for group in groups] == ["shared", "audio"]
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["lr"] == 1e-3


def test_geometry_regularization_zero_at_initialization_and_positive_after_change():
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4))

    assert geometry_regularization(model) == 0
    with torch.no_grad():
        model.bridge.gaussians.means.add_(1.0)

    assert geometry_regularization(model) > 0


def test_joint_loss_combines_visual_audio_and_regularization():
    pred_rgb = torch.zeros(1, 2, 2, 3)
    target_rgb = torch.ones(1, 2, 2, 3)
    pred_audio = torch.zeros(2, 1024)
    target_audio = torch.ones(2, 1024)
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4))

    losses = joint_loss(
        model=model,
        pred_rgb=pred_rgb,
        target_rgb=target_rgb,
        pred_audio=pred_audio,
        target_audio=target_audio,
        weights={"rgb_l1": 1.0, "audio_l1": 1.0, "geo": 0.1},
    )

    assert losses["total"] > 0
    assert losses["rgb_l1"] > 0
    assert losses["audio_l1"] > 0
