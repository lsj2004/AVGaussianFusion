import pytest
import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio.acoustic_gaussians import AcousticGaussianParameters
from avfusion.audio.losses import audiogs_mono_diff_loss, stft_magnitude_loss
from avfusion.audio.renderer import render_audio


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
