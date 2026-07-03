import torch
import pytest

from avfusion.adapters.visual_to_acoustic import select_topk_acoustic_carrier
from avfusion.visual.carrier import FrozenVisualCarrier


def test_select_topk_acoustic_carrier_orders_by_opacity():
    carrier = FrozenVisualCarrier(
        means=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        ),
        scales=torch.zeros(5, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(5, 1),
        opacities=torch.tensor([[0.0], [4.0], [-1.0], [2.0], [3.0]]),
        times=torch.zeros(5, 1),
        durations=torch.zeros(5, 1),
        velocities=torch.zeros(5, 3),
        max_duration=float("inf"),
    )

    acoustic = select_topk_acoustic_carrier(carrier, t=0.0, top_k=3)

    assert acoustic.visual_indices.tolist() == [1, 4, 3]
    assert torch.allclose(
        acoustic.opacity,
        torch.sigmoid(torch.tensor([[4.0], [3.0], [2.0]])),
    )
    assert torch.allclose(
        acoustic.xyz,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
    )
    assert acoustic.visual_indices.device.type == "cpu"
    assert not acoustic.xyz.requires_grad
    assert not acoustic.opacity.requires_grad


def test_select_topk_acoustic_carrier_handles_k_bounds():
    carrier = FrozenVisualCarrier(
        means=torch.arange(9, dtype=torch.float32).reshape(3, 3),
        scales=torch.zeros(3, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        opacities=torch.tensor([[0.0], [1.0], [2.0]]),
        times=torch.zeros(3, 1),
        durations=torch.zeros(3, 1),
        velocities=torch.zeros(3, 3),
        max_duration=float("inf"),
    )

    empty = select_topk_acoustic_carrier(carrier, t=0.0, top_k=0)
    assert empty.xyz.shape == (0, 3)
    assert empty.opacity.shape == (0, 1)
    assert empty.visual_indices.tolist() == []

    all_points = select_topk_acoustic_carrier(carrier, t=0.0, top_k=99)
    assert all_points.visual_indices.tolist() == [2, 1, 0]

    with pytest.raises(ValueError, match="top_k"):
        select_topk_acoustic_carrier(carrier, t=0.0, top_k=-1)


def test_select_topk_acoustic_carrier_stably_orders_ties_and_copies():
    carrier = FrozenVisualCarrier(
        means=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        scales=torch.zeros(4, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
        opacities=torch.ones(4, 1),
        times=torch.zeros(4, 1),
        durations=torch.zeros(4, 1),
        velocities=torch.zeros(4, 3),
        max_duration=float("inf"),
    )

    acoustic = select_topk_acoustic_carrier(carrier, t=0.0, top_k=3)
    acoustic.xyz.add_(100)
    state = carrier.query(0.0)

    assert acoustic.visual_indices.tolist() == [0, 1, 2]
    assert torch.allclose(state.xyz, torch.arange(12, dtype=torch.float32).reshape(4, 3))
