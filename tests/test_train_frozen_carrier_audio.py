import pytest
import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio import AcousticGaussianParameters
from avfusion.train.train_frozen_carrier_audio import main, train_one_step
from avfusion.visual.carrier import FrozenVisualCarrier


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


def test_main_parses_args_and_exits_with_placeholder():
    with pytest.raises(SystemExit, match="carrier export/checkpoint is not available yet"):
        main(
            [
                "--manifest",
                "runs/scene1_opera_a/scene_manifest.json",
                "--carrier",
                "runs/scene1_opera_a/carrier.pt",
                "--steps",
                "1000",
                "--lr",
                "0.0005",
            ]
        )
