import pytest
import torch

from avfusion.audio.losses import audiogs_mono_diff_loss


def test_audiogs_mono_diff_loss_uses_diff_weight():
    sample_count = 4096
    target = torch.zeros(2, sample_count)
    pred = torch.zeros(2, sample_count)
    pred[0] = 0.25
    pred[1] = -0.25

    low = audiogs_mono_diff_loss(pred, target, diff_weight=0.1)
    high = audiogs_mono_diff_loss(pred, target, diff_weight=2.0)

    assert high > low


def test_audiogs_mono_diff_loss_rejects_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        audiogs_mono_diff_loss(torch.zeros(1, 1024), torch.zeros(1, 1024))
