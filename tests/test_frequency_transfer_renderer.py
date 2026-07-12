import torch

from avfusion.soft.acoustic_field import AcousticGaussianField
from avfusion.soft.frequency_transfer_renderer import FrequencyTransferRenderer


def _visual_state(num_points=5):
    return {
        "xyz": torch.randn(num_points, 3),
        "opacity": torch.zeros(num_points, 1),
        "velocity": torch.zeros(num_points, 3),
    }


def test_frequency_transfer_renderer_outputs_stereo_audio_and_transfer_curve():
    field = AcousticGaussianField.from_visual_state(_visual_state(), num_points=6)
    renderer = FrequencyTransferRenderer(n_fft=512, hop_length=160, win_length=400)
    source = torch.randn(2, 2048)

    pred, debug = renderer(field.query(torch.tensor([[0.0]])), source, return_debug=True)

    assert pred.shape == (2, 2048)
    assert torch.isfinite(pred).all()
    assert debug["left_transfer"].shape == (257,)
    assert debug["right_transfer"].shape == (257,)
    assert debug["left_transfer_tf"].shape[-2] == 257
    assert debug["left_transfer_tf"].shape[-1] > 1


def test_frequency_transfer_renderer_backpropagates_to_acoustic_attributes():
    field = AcousticGaussianField.from_visual_state(_visual_state(), num_points=6)
    renderer = FrequencyTransferRenderer(n_fft=512, hop_length=160, win_length=400)
    source = torch.randn(2, 2048)

    loss = renderer(field.query(torch.tensor([[0.0]])), source).pow(2).mean()
    loss.backward()

    assert field.means.grad is not None
    assert field.means.grad.abs().sum() > 0
    assert field.mono_response.grad is not None
    assert field.mono_response.grad.abs().sum() > 0
    assert field.diff_response.grad is not None
    assert field.diff_response.grad.abs().sum() > 0
    assert field.distance_decay.grad is not None


def test_frequency_transfer_renderer_rejects_wrong_frequency_bins():
    field = AcousticGaussianField.from_visual_state(_visual_state(), num_points=6, num_frequency_bins=129)
    renderer = FrequencyTransferRenderer(n_fft=512, hop_length=160, win_length=400)

    try:
        renderer(field.query(torch.tensor([[0.0]])), torch.randn(2, 2048))
    except ValueError as exc:
        assert "frequency" in str(exc)
    else:
        raise AssertionError("mismatched frequency bins should raise ValueError")
