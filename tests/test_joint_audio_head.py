import torch

from avfusion.joint.audio_head import JointAudioHead


def _state(num_points=5):
    return {
        "xyz": torch.randn(num_points, 3, requires_grad=True),
        "opacity": torch.zeros(num_points, 1, requires_grad=True),
        "velocity": torch.zeros(num_points, 3, requires_grad=True),
    }


def test_joint_audio_head_outputs_stereo_audio():
    head = JointAudioHead(num_points=5)
    source = torch.randn(2, 1024)

    pred = head(_state(), source)

    assert pred.shape == (2, 1024)


def test_joint_audio_head_backpropagates_to_audio_params_and_geometry():
    head = JointAudioHead(num_points=5)
    state = _state()
    source = torch.randn(2, 1024)

    loss = head(state, source).pow(2).mean()
    loss.backward()

    assert head.mono_gain.grad is not None
    assert head.diff_gain.grad is not None
    assert state["xyz"].grad is not None
    assert state["opacity"].grad is not None


def test_joint_audio_head_top_k_limits_points():
    head = JointAudioHead(num_points=5, top_k=3)

    assert head.active_count == 3
