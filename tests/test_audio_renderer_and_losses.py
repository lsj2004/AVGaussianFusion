import pytest
import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio.acoustic_gaussians import AcousticGaussianParameters
from avfusion.audio.losses import stft_magnitude_loss
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
