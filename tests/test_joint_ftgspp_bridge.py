import pytest
import torch

from avfusion.joint.ftgspp_bridge import (
    FTGSDependencyError,
    FTGSRendererBridge,
)


class FakeGaussians(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.means = torch.nn.Parameter(torch.zeros(4, 3))
        self.opacities = torch.nn.Parameter(torch.zeros(4, 1))
        self.velocity_model = torch.nn.Parameter(torch.zeros(4, 3))
        self.sh_degree = 0

    def forward(self, t, w2c, intrinsic, shape, sh_degree=None, clamp=True):
        batch = w2c.shape[0]
        image = self.means.sum() * torch.ones(batch, shape[0], shape[1], 3)
        alpha = torch.ones(batch, shape[0], shape[1], 1)
        return image, alpha, {"means2d": torch.zeros(batch, 4, 2)}

    def means_t(self, t):
        return self.means + t.reshape(1, 1) * self.velocity_model

    def opacities_t(self, t):
        return self.opacities.sigmoid()


class MissingVelocityGaussians(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.means = torch.nn.Parameter(torch.zeros(4, 3))

    def forward(self, t, w2c, intrinsic, shape, sh_degree=None, clamp=True):
        return torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1, 1), {}

    def means_t(self, t):
        return self.means

    def opacities_t(self, t):
        return torch.ones(4, 1)


def test_bridge_render_rgb_calls_gaussian_forward():
    bridge = FTGSRendererBridge(FakeGaussians())
    batch = {
        "time": torch.tensor([[0.0]]),
        "w2c": torch.eye(4).reshape(1, 4, 4),
        "intrinsic": torch.eye(3).reshape(1, 3, 3),
        "height": 2,
        "width": 3,
    }

    image = bridge.render_rgb(batch)

    assert image.shape == (1, 2, 3, 3)
    assert image.requires_grad


def test_bridge_render_rgb_rejects_mixed_time_batch():
    bridge = FTGSRendererBridge(FakeGaussians())
    batch = {
        "time": torch.tensor([[0.0], [0.5]]),
        "w2c": torch.eye(4).reshape(1, 4, 4).repeat(2, 1, 1),
        "intrinsic": torch.eye(3).reshape(1, 3, 3).repeat(2, 1, 1),
        "height": 2,
        "width": 3,
    }

    with pytest.raises(ValueError, match="same time|mixed"):
        bridge.render_rgb(batch)


def test_bridge_query_state_keeps_gradient_to_shared_geometry():
    bridge = FTGSRendererBridge(FakeGaussians())

    state = bridge.query_state(torch.tensor([[0.5]]))
    loss = state["xyz"].sum() + state["opacity"].sum()
    loss.backward()

    assert bridge.gaussians.means.grad is not None
    assert bridge.gaussians.opacities.grad is not None


def test_load_checkpoint_reports_missing_ftgspp_dependency(monkeypatch, tmp_path):
    def fake_import(name):
        raise ImportError("missing ftgspp")

    monkeypatch.setattr("importlib.import_module", fake_import)

    with pytest.raises(FTGSDependencyError, match="FTGS\\+\\+|tinycudann|CUDA"):
        FTGSRendererBridge.load_checkpoint(tmp_path / "gaussians.pt")


def test_load_checkpoint_reports_tinycudann_runtime_errors(monkeypatch, tmp_path):
    def fake_import(name):
        raise OSError("Unknown compute capability")

    monkeypatch.setattr("importlib.import_module", fake_import)

    with pytest.raises(FTGSDependencyError, match="tinycudann|CUDA|FTGS\\+\\+"):
        FTGSRendererBridge.load_checkpoint(tmp_path / "gaussians.pt")


def test_load_checkpoint_rejects_unsupported_payload_with_clear_error(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "unsupported.pt"
    torch.save({"not_gaussians": torch.zeros(1)}, checkpoint_path)
    monkeypatch.setattr("importlib.import_module", lambda name: object())

    with pytest.raises(TypeError, match='raw FTGS\\+\\+ Gaussians module|dict with "gaussians"'):
        FTGSRendererBridge.load_checkpoint(checkpoint_path)


def test_load_checkpoint_rejects_payload_without_velocity_surface(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "missing_velocity.pt"
    torch.save(MissingVelocityGaussians(), checkpoint_path)
    monkeypatch.setattr("importlib.import_module", lambda name: object())

    with pytest.raises(TypeError, match="velocity|rendering|dict"):
        FTGSRendererBridge.load_checkpoint(checkpoint_path)
