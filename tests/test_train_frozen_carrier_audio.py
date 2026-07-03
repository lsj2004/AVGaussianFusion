import pytest
import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio import AcousticGaussianParameters
from avfusion.train.train_frozen_carrier_audio import main, train_audio_parameters, train_one_step
from avfusion.visual.carrier import FrozenVisualCarrier
from tests.test_audio_video_dataset import _write_manifest


def test_train_one_step_optimizes_audio_without_mutating_visual_carrier():
    carrier = FrozenVisualCarrier(
        means=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        scales=torch.zeros(4, 3),
        quats=torch.zeros(4, 4),
        opacities=torch.ones(4, 1),
        times=torch.zeros(4, 1),
        durations=torch.zeros(4, 1),
        velocities=torch.zeros(4, 3),
        max_duration=1.0,
    )
    original_means = carrier.means.clone()
    acoustic = AcousticCarrier(
        xyz=carrier.means.detach().clone(),
        opacity=torch.ones(4, 1),
        visual_indices=torch.arange(4),
    )
    source = torch.linspace(-1.0, 1.0, 2048).repeat(2, 1)
    target = source * 0.5
    params = AcousticGaussianParameters(num_points=4)
    before = params.mono_gain.detach().clone()

    loss, updated = train_one_step(acoustic, source, target, lr=0.01, params=params)

    assert loss > 0
    assert updated is params
    assert not torch.allclose(params.mono_gain.detach(), before)
    assert torch.allclose(carrier.means, original_means)
    carrier.assert_frozen()


def _write_carrier(path):
    carrier = FrozenVisualCarrier(
        means=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        scales=torch.zeros(4, 3),
        quats=torch.zeros(4, 4),
        opacities=torch.ones(4, 1),
        times=torch.zeros(4, 1),
        durations=torch.zeros(4, 1),
        velocities=torch.zeros(4, 3),
        max_duration=1.0,
    )
    carrier.save(path)


def test_train_audio_parameters_saves_checkpoint(tmp_path):
    manifest = _write_manifest(tmp_path)
    carrier_path = tmp_path / "carrier.pt"
    checkpoint_path = tmp_path / "stage2_audio.pt"
    _write_carrier(carrier_path)

    summary = train_audio_parameters(
        manifest_path=manifest,
        carrier_path=carrier_path,
        output_path=checkpoint_path,
        steps=2,
        lr=0.01,
        top_k=3,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert summary["steps"] == 2
    assert summary["train_cameras"] == ["cam00"]
    assert checkpoint["top_k"] == 3
    assert checkpoint["params"]["mono_gain"].shape == (3, 1)
    assert checkpoint["loss_history"]


def test_main_trains_and_writes_checkpoint(tmp_path):
    manifest = _write_manifest(tmp_path)
    carrier_path = tmp_path / "carrier.pt"
    checkpoint_path = tmp_path / "stage2_audio.pt"
    _write_carrier(carrier_path)

    main(
        [
            "--manifest",
            str(manifest),
            "--carrier",
            str(carrier_path),
            "--output",
            str(checkpoint_path),
            "--steps",
            "1",
            "--lr",
            "0.01",
            "--top-k",
            "2",
        ]
    )

    assert checkpoint_path.exists()
