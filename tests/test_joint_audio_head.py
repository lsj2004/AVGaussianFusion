import torch

from avfusion.joint.audio_head import JointAudioHead, SpectralJointAudioHead


def _state(num_points=5):
    return {
        "xyz": torch.randn(num_points, 3, requires_grad=True),
        "opacity": torch.zeros(num_points, 1, requires_grad=True),
        "velocity": torch.zeros(num_points, 3, requires_grad=True),
    }


def _assert_nonzero_grad(param):
    assert param.grad is not None
    assert param.grad.abs().sum() > 0


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

    _assert_nonzero_grad(head.audio_opacity)
    _assert_nonzero_grad(head.mono_gain)
    _assert_nonzero_grad(head.diff_gain)
    _assert_nonzero_grad(head.delay_offset)
    _assert_nonzero_grad(head.attenuation_logit)
    assert state["xyz"].grad is not None
    assert state["xyz"].grad.abs().sum() > 0
    assert state["opacity"].grad is not None
    assert state["opacity"].grad.abs().sum() > 0
    assert state["velocity"].grad is not None
    assert state["velocity"].grad.abs().sum() > 0


def test_joint_audio_head_top_k_limits_points():
    head = JointAudioHead(num_points=5, top_k=3)

    assert head.active_count == 3


def test_joint_audio_head_keeps_output_finite_for_huge_velocity():
    head = JointAudioHead(num_points=5)
    state = _state()
    velocity = torch.zeros(5, 3)
    velocity[0] = torch.tensor([1e30, 1e30, 1e30])
    state["velocity"] = velocity.requires_grad_()
    source = torch.randn(2, 1024)

    pred = head(state, source)

    assert torch.isfinite(pred).all()


def test_joint_audio_head_rejects_invalid_top_k():
    for top_k in (1, 0, -1):
        try:
            JointAudioHead(num_points=5, top_k=top_k)
        except ValueError as exc:
            assert "top_k" in str(exc)
        else:
            raise AssertionError(f"top_k={top_k} should raise ValueError")


def test_joint_audio_head_requires_at_least_two_points_for_route_weighting():
    try:
        JointAudioHead(num_points=1)
    except ValueError as exc:
        assert "at least two points" in str(exc)
    else:
        raise AssertionError("num_points=1 should raise ValueError")


def test_spectral_joint_audio_head_outputs_stereo_audio_with_stft_grid():
    head = SpectralJointAudioHead(num_points=5, num_frequency_bins=257, top_k=3)
    source = torch.randn(2, 2048)

    pred = head(_state(), source)

    assert pred.shape == (2, 2048)
    assert torch.isfinite(pred).all()
    assert head.active_count == 3


def test_spectral_joint_audio_head_backpropagates_to_masks_and_geometry():
    head = SpectralJointAudioHead(num_points=5, num_frequency_bins=257, top_k=4)
    state = _state()
    source = torch.randn(2, 2048)

    loss = head(state, source).pow(2).mean()
    loss.backward()

    _assert_nonzero_grad(head.audio_opacity)
    _assert_nonzero_grad(head.mono_mask)
    _assert_nonzero_grad(head.diff_mask)
    _assert_nonzero_grad(head.distance_logit)
    assert state["xyz"].grad is not None
    assert state["xyz"].grad.abs().sum() > 0
    assert state["opacity"].grad is not None
    assert state["opacity"].grad.abs().sum() > 0
    assert state["velocity"].grad is not None
    assert state["velocity"].grad.abs().sum() > 0
