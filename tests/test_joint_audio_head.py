import torch
import pytest

from avfusion.joint.audio_head import (
    AudioGSMaskedSpectralHead,
    JointAudioHead,
    SpectralJointAudioHead,
)


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


def test_audiogs_masked_spectral_head_outputs_stereo_audio():
    head = AudioGSMaskedSpectralHead(num_points=5, num_frequency_bins=257, top_k=3)
    source = torch.randn(2, 2048)

    pred = head(_state(), source)

    assert pred.shape == (2, 2048)
    assert torch.isfinite(pred).all()
    assert head.active_count == 3


def test_audiogs_masked_spectral_head_can_use_dual_branch_unet_renderer():
    head = AudioGSMaskedSpectralHead(
        num_points=5,
        num_frequency_bins=257,
        top_k=3,
        renderer_type="unet",
        use_stereo_cues=True,
        diff_use_inv_distance=True,
        diff_use_side_mag=True,
    )
    state = _state()
    source = torch.randn(2, 4096)

    pred = head(state, source)
    loss = pred.pow(2).mean()
    loss.backward()

    assert pred.shape == (2, 4096)
    assert torch.isfinite(pred).all()
    assert head.renderer is not None
    assert head.renderer.diff_in_channels == 4
    _assert_nonzero_grad(head.renderer.out_mono.weight)
    _assert_nonzero_grad(head.renderer.out_diff.weight)
    _assert_nonzero_grad(head.mono_sh)
    _assert_nonzero_grad(head.diff_sh)
    assert state["xyz"].grad is not None
    assert state["xyz"].grad.abs().sum() > 0


def test_audiogs_masked_spectral_head_uses_degree3_sh_basis_by_default():
    head = AudioGSMaskedSpectralHead(num_points=5, num_frequency_bins=257)

    assert head.sh_degree == 3
    assert head.sh_basis_dim == 16
    assert head.mono_sh.shape == (5, 257, 16)
    assert head.diff_sh.shape == (5, 257, 16)


def test_audiogs_masked_spectral_head_only_allocates_audio_params_for_top_k_carrier():
    head = AudioGSMaskedSpectralHead(num_points=100, num_frequency_bins=257, top_k=7)

    assert head.active_count == 7
    assert head.audio_opacity.shape == (7, 1)
    assert head.rotation.shape == (7, 4)
    assert head.freq_atten_logit.shape == (7, 257)
    assert head.mono_sh.shape == (7, 257, 16)
    assert head.diff_sh.shape == (7, 257, 16)


def test_audiogs_masked_spectral_head_initializes_identity_point_quaternions():
    head = AudioGSMaskedSpectralHead(num_points=5, num_frequency_bins=257, top_k=3)

    assert head.rotation[:, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert head.rotation[:, 1:].abs().sum().item() == pytest.approx(0.0)


def test_audiogs_masked_spectral_head_uses_standard_sh_constants():
    head = AudioGSMaskedSpectralHead(num_points=5, num_frequency_bins=257, top_k=3)
    xyz = torch.tensor([[0.0, 0.0, 1.0]])
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    features, _ = head._direction_features(xyz, rotation)

    assert features.shape == (1, 16)
    assert features[0, 0].item() == pytest.approx(0.28209479177387814)
    assert features[0, 1].item() == pytest.approx(0.0)
    assert features[0, 2].item() == pytest.approx(0.4886025119029199)
    assert features[0, 6].item() == pytest.approx(0.6307831305050401)
    assert features[0, 12].item() == pytest.approx(0.7463526651802308)


def test_audiogs_masked_spectral_head_rotates_directions_into_point_local_frame():
    head = AudioGSMaskedSpectralHead(num_points=5, num_frequency_bins=257, top_k=3)
    xyz = torch.tensor([[1.0, 0.0, 0.0]])
    sqrt_half = 2.0**-0.5
    local_rot_z90 = torch.tensor([[sqrt_half, 0.0, 0.0, sqrt_half]])

    features, _ = head._direction_features(xyz, local_rot_z90)

    assert features[0, 1].item() == pytest.approx(-0.4886025119029199)
    assert features[0, 2].item() == pytest.approx(0.0)
    assert features[0, 3].item() == pytest.approx(0.0, abs=1e-6)


def test_audiogs_masked_spectral_head_loads_legacy_three_vector_rotation_state():
    head = AudioGSMaskedSpectralHead(num_points=5, num_frequency_bins=257, top_k=3)
    state_dict = head.state_dict()
    state_dict["rotation"] = torch.zeros(3, 3)

    head.load_state_dict(state_dict)

    assert head.rotation.shape == (3, 4)
    assert head.rotation[:, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert head.rotation[:, 1:].abs().sum().item() == pytest.approx(0.0)


def test_audiogs_masked_spectral_head_uses_camera_frame_direction_features():
    head = AudioGSMaskedSpectralHead(num_points=5, num_frequency_bins=257, top_k=3)
    xyz = torch.tensor([[1.0, 0.0, 0.0]])
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    identity_w2c = torch.eye(4).reshape(1, 4, 4)
    rot_z_w2c = torch.tensor(
        [
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )

    identity_features, identity_distance = head._direction_features(
        xyz,
        rotation,
        camera_w2c=identity_w2c,
    )
    rotated_features, rotated_distance = head._direction_features(
        xyz,
        rotation,
        camera_w2c=rot_z_w2c,
    )

    assert identity_distance.item() == pytest.approx(1.0)
    assert rotated_distance.item() == pytest.approx(1.0)
    assert identity_features[0, 1:4].tolist() == pytest.approx([0.0, 0.0, -0.4886025119029199])
    assert rotated_features[0, 1:4].tolist() == pytest.approx([-0.4886025119029199, 0.0, 0.0])


def test_audiogs_masked_spectral_head_computes_geometry_phase_from_itd():
    head = AudioGSMaskedSpectralHead(
        num_points=5,
        num_frequency_bins=257,
        top_k=3,
        sample_rate=16000,
        use_geom_phase=True,
        head_radius=0.0875,
        sound_speed=343.0,
    )
    directions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    weights = torch.ones(2, 257) / 2.0

    phase = head._geometry_phase_delta(directions, weights)

    assert phase.shape == (257, 1)
    assert phase[0].item() == pytest.approx(0.0)
    assert phase.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    one_sided = head._geometry_phase_delta(directions[:1], weights[:1])
    assert one_sided.shape == (257, 1)
    assert one_sided[32].abs().item() > 0.0


def test_audiogs_masked_spectral_head_computes_mirrored_ear_distance_gains():
    head = AudioGSMaskedSpectralHead(
        num_points=5,
        num_frequency_bins=257,
        top_k=3,
        head_radius=0.1,
        use_ear_distance_attenuation=True,
    )
    camera_xyz = torch.tensor([[1.0, 0.0, 0.0]])
    weights = torch.ones(1, 257)

    left_gain, right_gain = head._ear_distance_gains(camera_xyz, weights)
    mirrored_left_gain, mirrored_right_gain = head._ear_distance_gains(-camera_xyz, weights)

    assert left_gain.shape == (257, 1)
    assert right_gain.shape == (257, 1)
    assert left_gain.mean().item() > right_gain.mean().item()
    assert mirrored_right_gain.mean().item() > mirrored_left_gain.mean().item()


def test_audiogs_masked_spectral_head_backpropagates_to_sh_masks_and_geometry():
    head = AudioGSMaskedSpectralHead(num_points=5, num_frequency_bins=257, top_k=4)
    state = _state()
    source = torch.randn(2, 2048)

    loss = head(state, source).pow(2).mean()
    loss.backward()

    _assert_nonzero_grad(head.audio_opacity)
    _assert_nonzero_grad(head.mono_sh)
    _assert_nonzero_grad(head.diff_sh)
    _assert_nonzero_grad(head.freq_atten_logit)
    _assert_nonzero_grad(head.rotation)
    assert state["xyz"].grad is not None
    assert state["xyz"].grad.abs().sum() > 0
    assert state["opacity"].grad is not None
    assert state["opacity"].grad.abs().sum() > 0
    assert state["velocity"].grad is not None
    assert state["velocity"].grad.abs().sum() > 0
