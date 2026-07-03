import torch

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
