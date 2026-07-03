import torch

from avfusion.visual.carrier import FrozenVisualCarrier


def test_carrier_query_uses_velocity_and_temporal_opacity():
    carrier = FrozenVisualCarrier(
        means=torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        scales=torch.zeros(2, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        opacities=torch.tensor([[0.0], [2.0]]),
        times=torch.tensor([[0.0], [0.5]]),
        durations=torch.zeros(2, 1),
        velocities=torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        max_duration=float("inf"),
    )

    state = carrier.query(0.5)

    assert torch.allclose(state.xyz[0], torch.tensor([1.5, 0.0, 0.0]))
    assert torch.allclose(state.xyz[1], torch.tensor([0.0, 2.0, 0.0]))
    assert state.opacity.shape == (2, 1)
    assert not state.xyz.requires_grad
    assert not state.opacity.requires_grad


def test_carrier_asserts_frozen_tensors():
    carrier = FrozenVisualCarrier(
        means=torch.zeros(1, 3),
        scales=torch.zeros(1, 3),
        quats=torch.ones(1, 4),
        opacities=torch.zeros(1, 1),
        times=torch.zeros(1, 1),
        durations=torch.zeros(1, 1),
        velocities=torch.zeros(1, 3),
        max_duration=float("inf"),
    )

    carrier.assert_frozen()


def test_carrier_save_load_round_trips(tmp_path):
    carrier = FrozenVisualCarrier(
        means=torch.tensor([[1.0, 2.0, 3.0]]),
        scales=torch.ones(1, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opacities=torch.tensor([[0.25]]),
        times=torch.tensor([[0.25]]),
        durations=torch.tensor([[0.5]]),
        velocities=torch.tensor([[0.0, 1.0, 0.0]]),
        max_duration=1.0,
    )
    path = tmp_path / "carrier.pt"

    carrier.save(path)
    loaded = FrozenVisualCarrier.load(path)

    assert torch.allclose(loaded.means, carrier.means)
    assert torch.allclose(loaded.velocities, carrier.velocities)
    assert loaded.max_duration == carrier.max_duration
    loaded.assert_frozen()
