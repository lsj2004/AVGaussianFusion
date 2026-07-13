import pytest
import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio.acoustic_gaussians import AcousticGaussianParameters
from avfusion.audio.losses import audiogs_mono_diff_loss, audio_spatial_loss, stft_magnitude_loss
from avfusion.audio.renderer import render_audio
from avfusion.soft.frequency_transfer_renderer import FrequencyTransferRenderer


def test_renderer_outputs_stereo_waveform_and_optimizable_params():
    carrier = AcousticCarrier(
        xyz=torch.zeros(4, 3),
        opacity=torch.ones(4, 1),
        visual_indices=torch.arange(4),
    )
    params = AcousticGaussianParameters(num_points=4)
    source = torch.randn(2, 1024)

    pred = render_audio(carrier, params, source)
    pred.sum().backward()

    assert pred.shape == source.shape
    assert pred.requires_grad
    assert params.mono_gain.shape == (4, 1)
    assert params.diff_gain.shape == (4, 1)
    assert params.mono_gain.requires_grad
    assert params.diff_gain.requires_grad
    assert params.mono_gain.grad is not None
    assert params.diff_gain.grad is not None


def test_stft_magnitude_loss_is_scalar_and_differentiable():
    pred = torch.randn(2, 2048, requires_grad=True)
    target = torch.randn(2, 2048)

    loss = stft_magnitude_loss(pred, target, n_fft=256, hop_length=64)

    assert loss.ndim == 0
    loss.backward()
    assert pred.grad is not None


def test_stft_magnitude_loss_rejects_shape_mismatch():
    pred = torch.randn(2, 2048)
    target = torch.randn(1, 2048)

    with pytest.raises(ValueError, match="shape mismatch"):
        stft_magnitude_loss(pred, target)


def test_acoustic_parameters_validate_opacity_shape_and_count():
    params = AcousticGaussianParameters(num_points=4)

    with pytest.raises(ValueError, match="shape"):
        params.aggregate(torch.ones(4))

    with pytest.raises(ValueError, match="point count"):
        params.aggregate(torch.ones(3, 1))


def test_renderer_rejects_invalid_source_shape():
    carrier = AcousticCarrier(
        xyz=torch.zeros(4, 3),
        opacity=torch.ones(4, 1),
        visual_indices=torch.arange(4),
    )
    params = AcousticGaussianParameters(num_points=4)

    with pytest.raises(ValueError, match="source_audio"):
        render_audio(carrier, params, torch.randn(1024))

    with pytest.raises(ValueError, match="source_audio"):
        render_audio(carrier, params, torch.randn(1, 1024))


def test_stft_magnitude_loss_rejects_short_audio():
    pred = torch.randn(2, 128)
    target = torch.randn(2, 128)

    with pytest.raises(ValueError, match="shorter than n_fft"):
        stft_magnitude_loss(pred, target, n_fft=512)


def test_audiogs_mono_diff_loss_supports_mr_stft_and_phase_loss():
    pred = torch.randn(2, 4096, requires_grad=True)
    target = torch.randn(2, 4096)

    loss = audiogs_mono_diff_loss(
        pred,
        target,
        mr_stft_scales=((256, 64, 256), (512, 160, 400)),
        phase_loss_weight=0.05,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_audiogs_mono_diff_loss_penalizes_over_amplified_tf_diff_ratio():
    base = torch.sin(torch.linspace(0, 12.0, 4096))
    target = torch.stack([1.05 * base, 0.95 * base])
    pred_overdiff = torch.stack([1.4 * base, 0.6 * base]).requires_grad_(True)
    pred_matched = target.clone().requires_grad_(True)

    overdiff_loss = audiogs_mono_diff_loss(
        pred_overdiff,
        target,
        n_fft=256,
        hop_length=64,
        win_length=256,
        diff_weight=0.0,
        lre_loss_weight=0.0,
        tf_diff_ratio_loss_weight=0.1,
        tf_diff_ratio_margin_db=0.0,
    )
    matched_loss = audiogs_mono_diff_loss(
        pred_matched,
        target,
        n_fft=256,
        hop_length=64,
        win_length=256,
        diff_weight=0.0,
        lre_loss_weight=0.0,
        tf_diff_ratio_loss_weight=0.1,
        tf_diff_ratio_margin_db=0.0,
    )

    assert overdiff_loss > matched_loss
    overdiff_loss.backward()
    assert pred_overdiff.grad is not None
    assert pred_overdiff.grad.abs().sum() > 0


def test_audio_spatial_loss_penalizes_stereo_errors():
    base = torch.sin(torch.linspace(0, 24.0, 4096))
    target = torch.stack([1.2 * base, 0.8 * base])
    matched = target.clone().requires_grad_(True)
    swapped = torch.stack([0.8 * base, 1.2 * base]).requires_grad_(True)

    matched_loss = audio_spatial_loss(
        matched,
        target,
        band_lre_weight=1.0,
        coherence_weight=0.1,
        phase_diff_weight=0.1,
        energy_weight=0.1,
        n_fft=256,
        hop_length=64,
        win_length=256,
    )
    swapped_loss = audio_spatial_loss(
        swapped,
        target,
        band_lre_weight=1.0,
        coherence_weight=0.1,
        phase_diff_weight=0.1,
        energy_weight=0.1,
        n_fft=256,
        hop_length=64,
        win_length=256,
    )

    assert matched_loss < 1e-6
    assert swapped_loss > matched_loss + 0.1
    swapped_loss.backward()
    assert swapped.grad is not None
    assert swapped.grad.abs().sum() > 0


def test_audio_spatial_loss_penalizes_tf_ild_errors():
    base = torch.sin(torch.linspace(0, 24.0, 4096))
    target = torch.stack([1.2 * base, 0.8 * base])
    matched = target.clone().requires_grad_(True)
    swapped = torch.stack([0.8 * base, 1.2 * base]).requires_grad_(True)

    matched_loss = audio_spatial_loss(
        matched,
        target,
        tf_ild_weight=1.0,
        n_fft=256,
        hop_length=64,
        win_length=256,
    )
    swapped_loss = audio_spatial_loss(
        swapped,
        target,
        tf_ild_weight=1.0,
        n_fft=256,
        hop_length=64,
        win_length=256,
    )

    assert matched_loss < 1e-6
    assert swapped_loss > matched_loss + 0.1
    swapped_loss.backward()
    assert swapped.grad is not None
    assert swapped.grad.abs().sum() > 0


def test_audio_spatial_loss_penalizes_mono_diff_phase_errors():
    t = torch.linspace(0, 24.0, 4096)
    target_mid = torch.sin(t)
    target_side = 0.25 * torch.cos(t)
    target = torch.stack([target_mid + target_side, target_mid - target_side])
    pred_side = 0.25 * torch.sin(t)
    pred = torch.stack([target_mid + pred_side, target_mid - pred_side]).requires_grad_(True)
    matched = target.clone().requires_grad_(True)

    matched_loss = audio_spatial_loss(
        matched,
        target,
        mono_diff_phase_weight=1.0,
        n_fft=256,
        hop_length=64,
        win_length=256,
    )
    phase_loss = audio_spatial_loss(
        pred,
        target,
        mono_diff_phase_weight=1.0,
        n_fft=256,
        hop_length=64,
        win_length=256,
    )

    assert matched_loss < 1e-6
    assert phase_loss > matched_loss + 0.01
    phase_loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_frequency_transfer_renderer_preserves_source_side_channel():
    renderer = FrequencyTransferRenderer(n_fft=256, hop_length=64, win_length=256)
    state = {
        "xyz": torch.zeros(4, 3),
        "opacity": torch.zeros(4, 1),
        "audio_opacity": torch.zeros(4, 1),
        "mono_response": torch.zeros(4, renderer.num_frequency_bins),
        "diff_response": torch.zeros(4, renderer.num_frequency_bins),
        "distance_decay": torch.zeros(4, renderer.num_frequency_bins),
        "phase_delay": torch.zeros(4, renderer.num_frequency_bins),
    }
    base = torch.sin(torch.linspace(0, 24.0, 4096))
    source = torch.stack([1.25 * base, 0.75 * base])

    pred, debug = renderer(state, source, return_debug=True)

    assert pred.shape == source.shape
    assert not torch.allclose(pred[0], pred[1], atol=1e-4)
    assert torch.mean(torch.abs(pred - source)) < 1e-3
    assert "mid_transfer" in debug
    assert "side_transfer" in debug
    assert "diff_transfer" in debug


def test_frequency_transfer_renderer_uses_learnable_side_response():
    renderer = FrequencyTransferRenderer(n_fft=256, hop_length=64, win_length=256)
    base_state = {
        "xyz": torch.zeros(4, 3),
        "opacity": torch.zeros(4, 1),
        "audio_opacity": torch.zeros(4, 1),
        "mono_response": torch.zeros(4, renderer.num_frequency_bins),
        "diff_response": torch.zeros(4, renderer.num_frequency_bins),
        "distance_decay": torch.zeros(4, renderer.num_frequency_bins),
        "phase_delay": torch.zeros(4, renderer.num_frequency_bins),
    }
    reduced_side_state = dict(base_state)
    reduced_side_state["side_response"] = torch.full((4, renderer.num_frequency_bins), -2.0)
    base = torch.sin(torch.linspace(0, 24.0, 4096))
    source = torch.stack([1.25 * base, 0.75 * base])

    original_side = renderer(base_state, source)[0] - renderer(base_state, source)[1]
    reduced_side = renderer(reduced_side_state, source)[0] - renderer(reduced_side_state, source)[1]

    assert reduced_side.abs().mean() < original_side.abs().mean()


def test_frequency_transfer_renderer_geometry_diff_head_modulates_stereo_from_source_ild():
    renderer = FrequencyTransferRenderer(
        n_fft=256,
        hop_length=64,
        win_length=256,
        use_geometry_diff_head=True,
    )
    with torch.no_grad():
        renderer.geometry_diff_head.weight.zero_()
        renderer.geometry_diff_head.bias.zero_()
        renderer.geometry_diff_head.weight[0, 3] = 0.5
    state = {
        "xyz": torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 2.0], [1.0, 0.0, 3.0], [1.0, 0.0, 4.0]]),
        "opacity": torch.zeros(4, 1),
        "audio_opacity": torch.zeros(4, 1),
        "mono_response": torch.zeros(4, renderer.num_frequency_bins),
        "diff_response": torch.zeros(4, renderer.num_frequency_bins),
        "distance_decay": torch.zeros(4, renderer.num_frequency_bins),
        "phase_delay": torch.zeros(4, renderer.num_frequency_bins),
    }
    base = torch.sin(torch.linspace(0, 24.0, 4096))
    left_louder = torch.stack([1.25 * base, 0.75 * base])
    right_louder = torch.stack([0.75 * base, 1.25 * base])

    _, left_debug = renderer(state, left_louder, return_debug=True)
    _, right_debug = renderer(state, right_louder, return_debug=True)

    assert left_debug["geometry_diff_delta"].mean() > 0
    assert right_debug["geometry_diff_delta"].mean() < 0
    assert not torch.allclose(left_debug["diff_transfer"], right_debug["diff_transfer"])
