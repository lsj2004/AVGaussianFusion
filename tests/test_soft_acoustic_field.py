import torch

from avfusion.soft.acoustic_field import AcousticGaussianField
from avfusion.soft.losses import soft_coupling_losses


def _visual_state(num_points=5):
    xyz = torch.arange(num_points * 3, dtype=torch.float32).reshape(num_points, 3) / 10.0
    return {
        "xyz": xyz,
        "opacity": torch.linspace(0.1, 0.9, num_points).reshape(num_points, 1),
        "velocity": torch.ones(num_points, 3) * 0.25,
    }


def test_acoustic_field_queries_independent_trainable_state():
    field = AcousticGaussianField.from_visual_state(
        _visual_state(),
        num_points=6,
        anchored_fraction=0.5,
        dynamic_fraction=0.25,
    )

    state = field.query(torch.tensor([[0.5]]))
    loss = (
        state["xyz"].sum()
        + state["opacity"].sum()
        + state["velocity"].sum()
        + state["mono_response"].sum()
        + state["diff_response"].sum()
        + state["diff_directional_response"].sum()
    )
    loss.backward()

    assert state["xyz"].shape == (6, 3)
    assert state["opacity"].shape == (6, 1)
    assert state["velocity"].shape == (6, 3)
    assert state["mono_response"].shape == (6, 257)
    assert state["diff_response"].shape == (6, 257)
    assert state["diff_directional_response"].shape == (6, 3, 257)
    assert state["distance_decay"].shape == (6, 257)
    assert state["phase_delay"].shape == (6, 257)
    assert state["audio_opacity"].shape == (6, 1)
    assert field.means.grad is not None
    assert field.opacities.grad is not None
    assert field.velocity_model.grad is not None
    assert field.mono_response.grad is not None
    assert field.diff_response.grad is not None
    assert field.diff_directional_response.grad is not None


def test_acoustic_field_initializes_directional_diff_basis():
    field = AcousticGaussianField.from_visual_state(_visual_state(), num_points=4, num_frequency_bins=8)
    state = field.query(torch.tensor([[0.0]]))

    assert state["diff_directional_response"].shape == (4, 3, 8)
    assert torch.allclose(state["diff_directional_response"], torch.zeros_like(state["diff_directional_response"]))


def test_acoustic_field_records_visual_anchor_groups():
    field = AcousticGaussianField.from_visual_state(
        _visual_state(),
        num_points=10,
        anchored_fraction=0.6,
        dynamic_fraction=0.2,
    )

    assert field.anchor_indices.shape == (10, 1)
    assert field.anchor_weights.shape == (10, 1)
    assert int(field.anchor_mask.sum().item()) == 8
    assert int(field.residual_mask.sum().item()) == 2


def test_soft_coupling_losses_use_stopgrad_visual_anchors():
    field = AcousticGaussianField.from_visual_state(_visual_state(), num_points=4)
    acoustic_state = field.query(torch.tensor([[0.0]]))
    visual_state = {key: value.clone().requires_grad_() for key, value in _visual_state().items()}

    losses = soft_coupling_losses(field, acoustic_state, visual_state)
    total = losses["anchor"] + losses["motion"] + losses["activity"] + losses["sparse"]
    total.backward()

    assert set(losses) == {"anchor", "motion", "activity", "sparse"}
    assert field.means.grad is not None
    assert visual_state["xyz"].grad is None
    assert visual_state["velocity"].grad is None
    assert visual_state["opacity"].grad is None


def test_acoustic_field_checkpoint_roundtrip(tmp_path):
    field = AcousticGaussianField.from_visual_state(_visual_state(), num_points=7)
    path = tmp_path / "acoustic_field.pt"

    field.save(path)
    loaded = AcousticGaussianField.load(path)

    assert torch.allclose(loaded.means, field.means)
    assert torch.allclose(loaded.mono_response, field.mono_response)
    assert torch.equal(loaded.anchor_indices, field.anchor_indices)
    assert torch.equal(loaded.anchor_mask, field.anchor_mask)
