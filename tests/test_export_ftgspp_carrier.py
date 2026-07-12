import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from avfusion.visual.carrier import FrozenVisualCarrier
from avfusion.visual.export_ftgspp_carrier import (
    carrier_from_ftgspp_object,
    export_checkpoint,
)


def _fake_gaussians(**overrides):
    data = {
        "means": torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        ),
        "scales": torch.zeros(2, 3, requires_grad=True),
        "quats": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], requires_grad=True
        ),
        "opacities": torch.zeros(2, 1, requires_grad=True),
        "times": torch.tensor([[0.0], [0.5]], requires_grad=True),
        "durations": torch.ones(2, 1, requires_grad=True),
        "velocity_model": torch.ones(2, 3, requires_grad=True),
        "max_duration": 2.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_carrier_from_ftgspp_object_freezes_required_fields():
    gs = _fake_gaussians()

    carrier = carrier_from_ftgspp_object(gs)

    assert isinstance(carrier, FrozenVisualCarrier)
    assert len(carrier) == 2
    assert torch.allclose(carrier.means, gs.means.detach())
    assert torch.allclose(carrier.velocities, torch.ones(2, 3))
    assert not carrier.means.requires_grad
    assert not carrier.velocities.requires_grad


def test_carrier_from_ftgspp_object_preserves_marginal_gates():
    gates = torch.tensor([[0.25], [-0.5]], requires_grad=True)
    gs = _fake_gaussians(marginal_gates=gates)

    carrier = carrier_from_ftgspp_object(gs)

    assert torch.allclose(carrier.marginal_gates, gates.detach())
    assert not carrier.marginal_gates.requires_grad


def test_carrier_from_ftgspp_object_rejects_velocity_field_like_object():
    gs = _fake_gaussians(velocity_model=SimpleNamespace())

    with pytest.raises(TypeError, match="velocity_model"):
        carrier_from_ftgspp_object(gs)


def test_export_checkpoint_uses_gaussians_key_and_saves_carrier(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "carrier.pt"
    gs = _fake_gaussians()
    torch.save({"gaussians": gs}, checkpoint_path)

    carrier = export_checkpoint(checkpoint_path, output_path)

    loaded = FrozenVisualCarrier.load(output_path)
    assert isinstance(carrier, FrozenVisualCarrier)
    assert torch.allclose(loaded.means, carrier.means)
    assert torch.allclose(loaded.velocities, torch.ones(2, 3))


def test_export_checkpoint_uses_loaded_object_without_gaussians_key(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    output_path = tmp_path / "carrier.pt"
    gs = _fake_gaussians()
    torch.save(gs, checkpoint_path)

    carrier = export_checkpoint(checkpoint_path, output_path)

    assert output_path.exists()
    assert torch.allclose(carrier.means, gs.means.detach())


def test_export_checkpoint_retries_with_tinycudann_stub_for_cpu_export(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "carrier.pt"
    gs = _fake_gaussians()
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise OSError(
                "Unknown compute capability. Ensure PyTorch with CUDA support is installed."
            )
        return {"gaussians": gs}

    monkeypatch.delitem(sys.modules, "tinycudann", raising=False)
    monkeypatch.delitem(sys.modules, "tinycudann.modules", raising=False)
    monkeypatch.setattr(torch, "load", fake_load)

    carrier = export_checkpoint(tmp_path / "checkpoint.pt", output_path)

    assert len(calls) == 2
    assert output_path.exists()
    assert torch.allclose(carrier.velocities, torch.ones(2, 3))
    assert hasattr(sys.modules["tinycudann"], "Encoding")


def test_export_script_runs_inside_ftgspp_environment():
    script = Path("scripts/export_stage1_carrier.sh").read_text()

    assert 'FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"' in script
    assert 'cd "${FTGSPP_ROOT}"' in script
    assert 'PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}"' in script
    assert "uv run --no-sync python -m avfusion.visual.export_ftgspp_carrier" in script


def test_scene7_export_scripts_use_current_carrier_module():
    for script_path in [
        Path("scripts/export_stage1_carrier_scene7_playing.sh"),
        Path("scripts/export_stage1_carrier_scene7_playing_300.sh"),
    ]:
        script = script_path.read_text()

        assert "python -m avfusion.visual.export_ftgspp_carrier" in script
        assert "avfusion.export.export_ftgspp_carrier" not in script
