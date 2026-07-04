import sys

import pytest
import torch

from avfusion.eval.audio_metrics import (
    compute_audiogs_metrics,
    compute_env_distance,
    compute_lre,
    compute_mag_distance,
)


def test_audio_metrics_are_zero_for_identical_stereo_waveforms():
    waveform = torch.stack(
        [
            torch.sin(torch.linspace(0, 4, 2048)),
            torch.cos(torch.linspace(0, 4, 2048)),
        ]
    )

    assert compute_mag_distance(waveform, waveform) == pytest.approx(0.0)
    assert compute_env_distance(waveform, waveform) == pytest.approx(0.0)
    assert compute_lre(waveform, waveform) == pytest.approx(0.0)


def test_lre_matches_left_right_energy_ratio_error():
    pred = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    assert compute_lre(pred, target) == pytest.approx(6.020578, rel=1e-5)


def test_compute_audiogs_metrics_reports_optional_dpam_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "cdpam", None)
    pred = torch.zeros(2, 2048)
    target = torch.ones(2, 2048)

    metrics = compute_audiogs_metrics(pred, target, sample_rate=16000, include_dpam=True)

    assert set(metrics) == {
        "MAG",
        "ENV",
        "LRE",
        "RTE",
        "RTE_available",
        "RTE_error",
        "DPAM",
        "DPAM_available",
        "DPAM_error",
    }
    assert metrics["MAG"] > 0
    assert metrics["ENV"] > 0
    assert metrics["LRE"] == pytest.approx(0.0)
    assert metrics["RTE"] is None
    assert metrics["RTE_available"] is False
    assert metrics["DPAM"] is None
    assert metrics["DPAM_available"] is False
    assert "cdpam" in metrics["DPAM_error"].lower()


def test_audio_metrics_reject_shape_mismatch_and_non_stereo():
    with pytest.raises(ValueError, match="same shape"):
        compute_audiogs_metrics(torch.zeros(2, 128), torch.zeros(2, 64), sample_rate=16000)

    with pytest.raises(ValueError, match="stereo"):
        compute_audiogs_metrics(torch.zeros(128), torch.zeros(128), sample_rate=16000)
