# Joint AV Gaussian Route B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Route B joint audio-visual Gaussian fine-tuning that reuses the FTGS++ differentiable renderer while preserving the existing Route A frozen-carrier baseline.

**Architecture:** Route B loads an FTGS++ `Gaussians` checkpoint, wraps it in a bridge that can render RGB through `Gaussians.forward(t, w2c, intrinsic, shape)`, and attaches a trainable geometry-aware audio head to the same Gaussian state. Training runs in two stages: audio warmup with shared Gaussians frozen, then joint fine-tuning with selected shared Gaussian parameters trainable under visual, audio, and geometry regularization losses.

**Tech Stack:** Python, PyTorch, FTGS++ `Gaussians`, `gsplat` through FTGS++, PyYAML, existing AVGaussianFusion audio/data/eval modules, pytest.

---

## File Structure

- Create `avfusion/joint/__init__.py`: public exports for Route B modules.
- Create `avfusion/joint/ftgspp_bridge.py`: FTGS++ import, checkpoint loading, RGB rendering wrapper, and clear dependency errors.
- Create `avfusion/joint/audio_head.py`: trainable audio head that consumes shared Gaussian state and source audio.
- Create `avfusion/joint/model.py`: `JointAVGaussianModel` that combines the FTGS++ bridge and audio head and exposes parameter groups.
- Create `avfusion/joint/losses.py`: visual/audio/reg loss helpers for joint training.
- Create `avfusion/train/train_joint_av_gaussians.py`: CLI and one-step/two-stage training entrypoint.
- Create `configs/scene1_opera_b_joint_av.yaml`: Route B defaults and paths.
- Create `scripts/train_joint_scene1_opera.sh`: launch Route B training in the FTGS++ environment.
- Create `scripts/eval_joint_scene1_opera.sh`: evaluate Route B outputs without overwriting Route A outputs.
- Create `tests/test_joint_ftgspp_bridge.py`: bridge import/loading/render adapter tests using fakes.
- Create `tests/test_joint_audio_head.py`: audio head shape, gradient, and parameter tests.
- Create `tests/test_joint_model_and_losses.py`: parameter grouping, regularization, and train-step tests.
- Create `tests/test_train_joint_av_gaussians.py`: CLI/config/checkpoint smoke tests.
- Modify `README.md`: add Route B commands while keeping Route A commands.

---

### Task 1: FTGS++ Bridge With Fakeable Renderer

**Files:**
- Create: `avfusion/joint/__init__.py`
- Create: `avfusion/joint/ftgspp_bridge.py`
- Test: `tests/test_joint_ftgspp_bridge.py`

- [ ] **Step 1: Write failing bridge tests**

Create `tests/test_joint_ftgspp_bridge.py`:

```python
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

    with pytest.raises(FTGSDependencyError, match="FTGS"):
        FTGSRendererBridge.load_checkpoint(tmp_path / "gaussians.pt")
```

- [ ] **Step 2: Run bridge tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with torch pytest tests/test_joint_ftgspp_bridge.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'avfusion.joint'`.

- [ ] **Step 3: Implement bridge module**

Create `avfusion/joint/__init__.py`:

```python
"""Joint audio-visual Gaussian route."""

from avfusion.joint.ftgspp_bridge import FTGSDependencyError, FTGSRendererBridge

__all__ = ["FTGSDependencyError", "FTGSRendererBridge"]
```

Create `avfusion/joint/ftgspp_bridge.py`:

```python
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class FTGSDependencyError(RuntimeError):
    pass


class FTGSRendererBridge:
    def __init__(self, gaussians: torch.nn.Module):
        self.gaussians = gaussians

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str | Path) -> "FTGSRendererBridge":
        try:
            importlib.import_module("ftgspp.models.gaussians")
            importlib.import_module("gsplat")
        except ImportError as error:
            raise FTGSDependencyError(
                "Route B joint training requires the FTGS++ environment with ftgspp and gsplat importable."
            ) from error
        gaussians = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(gaussians)

    def to(self, device: torch.device | str) -> "FTGSRendererBridge":
        self.gaussians.to(device)
        return self

    def render_rgb(self, batch: dict[str, Any], sh_degree: int | None = None) -> Tensor:
        image, _, _ = self.gaussians(
            t=batch["time"][0],
            w2c=batch["w2c"],
            intrinsic=batch["intrinsic"],
            shape=(int(batch["height"]), int(batch["width"])),
            sh_degree=sh_degree,
        )
        return image

    def query_state(self, t: Tensor) -> dict[str, Tensor]:
        return {
            "xyz": self.gaussians.means_t(t),
            "opacity": self.gaussians.opacities_t(t),
            "velocity": self._velocity_at(t),
        }

    def _velocity_at(self, t: Tensor) -> Tensor:
        velocity_model = getattr(self.gaussians, "velocity_model")
        if isinstance(velocity_model, torch.Tensor):
            return velocity_model
        return self.gaussians.velocities_t(t)
```

- [ ] **Step 4: Run bridge tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with torch pytest tests/test_joint_ftgspp_bridge.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit bridge**

Run:

```bash
git add avfusion/joint/__init__.py avfusion/joint/ftgspp_bridge.py tests/test_joint_ftgspp_bridge.py
git commit -m "feat: add FTGS renderer bridge"
```

---

### Task 2: Geometry-Aware Joint Audio Head

**Files:**
- Create: `avfusion/joint/audio_head.py`
- Test: `tests/test_joint_audio_head.py`

- [ ] **Step 1: Write failing audio head tests**

Create `tests/test_joint_audio_head.py`:

```python
import torch

from avfusion.joint.audio_head import JointAudioHead


def _state(num_points=5):
    return {
        "xyz": torch.randn(num_points, 3, requires_grad=True),
        "opacity": torch.zeros(num_points, 1, requires_grad=True),
        "velocity": torch.zeros(num_points, 3, requires_grad=True),
    }


def test_joint_audio_head_outputs_stereo_audio():
    head = JointAudioHead(num_points=5)
    source = torch.randn(2, 1024)

    pred = head(_state(), source)

    assert pred.shape == (2, 1024)


def test_joint_audio_head_backpropagates_to_audio_params_and_geometry():
    head = JointAudioHead(num_points=5)
    state = _state()
    source = torch.randn(2, 1024)

    loss = head(state, source).pow(2).mean()
    loss.backward()

    assert head.mono_gain.grad is not None
    assert head.diff_gain.grad is not None
    assert state["xyz"].grad is not None
    assert state["opacity"].grad is not None


def test_joint_audio_head_top_k_limits_points():
    head = JointAudioHead(num_points=5, top_k=3)

    assert head.active_count == 3
```

- [ ] **Step 2: Run audio head tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with torch pytest tests/test_joint_audio_head.py -v
```

Expected: import fails for `avfusion.joint.audio_head`.

- [ ] **Step 3: Implement audio head**

Create `avfusion/joint/audio_head.py`:

```python
from __future__ import annotations

import torch
from torch import Tensor, nn


class JointAudioHead(nn.Module):
    def __init__(self, num_points: int, top_k: int | None = None):
        super().__init__()
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        self.num_points = int(num_points)
        self.top_k = min(int(top_k), num_points) if top_k is not None else num_points
        self.audio_opacity = nn.Parameter(torch.zeros(num_points, 1))
        self.mono_gain = nn.Parameter(torch.zeros(num_points, 1))
        self.diff_gain = nn.Parameter(torch.zeros(num_points, 1))
        self.delay_offset = nn.Parameter(torch.zeros(num_points, 1))
        self.attenuation_logit = nn.Parameter(torch.zeros(num_points, 1))

    @property
    def active_count(self) -> int:
        return self.top_k

    def forward(self, state: dict[str, Tensor], source_audio: Tensor) -> Tensor:
        if source_audio.ndim != 2 or source_audio.shape[0] != 2:
            raise ValueError(f"source_audio must have shape (2, samples), got {tuple(source_audio.shape)}")
        xyz = state["xyz"]
        opacity = state["opacity"]
        scores = (opacity + self.audio_opacity).reshape(-1)
        indices = torch.argsort(scores, descending=True, stable=True)[: self.top_k]
        selected_opacity = scores.index_select(0, indices).reshape(-1, 1)
        weights = torch.softmax(selected_opacity, dim=0)
        selected_xyz = xyz.index_select(0, indices)
        distance = selected_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        attenuation = torch.sigmoid(self.attenuation_logit.index_select(0, indices)) / distance
        weights = weights * attenuation
        weights = weights / weights.sum().clamp_min(1e-6)
        mono = torch.tanh((weights * self.mono_gain.index_select(0, indices)).sum())
        diff = torch.tanh((weights * self.diff_gain.index_select(0, indices)).sum())
        mono_source = source_audio.to(mono).mean(dim=0, keepdim=True)
        left = mono_source * (1 + mono + diff)
        right = mono_source * (1 + mono - diff)
        return torch.cat([left, right], dim=0)
```

- [ ] **Step 4: Run audio head tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with torch pytest tests/test_joint_audio_head.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit audio head**

Run:

```bash
git add avfusion/joint/audio_head.py tests/test_joint_audio_head.py
git commit -m "feat: add joint audio head"
```

---

### Task 3: Joint Model And Loss Helpers

**Files:**
- Create: `avfusion/joint/model.py`
- Create: `avfusion/joint/losses.py`
- Modify: `avfusion/joint/__init__.py`
- Test: `tests/test_joint_model_and_losses.py`

- [ ] **Step 1: Write failing joint model/loss tests**

Create `tests/test_joint_model_and_losses.py`:

```python
import torch

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.losses import geometry_regularization, joint_loss
from avfusion.joint.model import JointAVGaussianModel
from tests.test_joint_ftgspp_bridge import FakeGaussians


def test_joint_model_exposes_separate_parameter_groups():
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4))

    groups = model.parameter_groups(shared_lr=1e-5, audio_lr=1e-3)

    assert [group["name"] for group in groups] == ["shared", "audio"]
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["lr"] == 1e-3


def test_geometry_regularization_zero_at_initialization_and_positive_after_change():
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4))

    assert geometry_regularization(model) == 0
    with torch.no_grad():
        model.bridge.gaussians.means.add_(1.0)

    assert geometry_regularization(model) > 0


def test_joint_loss_combines_visual_audio_and_regularization():
    pred_rgb = torch.zeros(1, 2, 2, 3)
    target_rgb = torch.ones(1, 2, 2, 3)
    pred_audio = torch.zeros(2, 1024)
    target_audio = torch.ones(2, 1024)
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4))

    losses = joint_loss(
        model=model,
        pred_rgb=pred_rgb,
        target_rgb=target_rgb,
        pred_audio=pred_audio,
        target_audio=target_audio,
        weights={"rgb_l1": 1.0, "audio_l1": 1.0, "geo": 0.1},
    )

    assert losses["total"] > 0
    assert losses["rgb_l1"] > 0
    assert losses["audio_l1"] > 0
```

- [ ] **Step 2: Run joint model tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with torch pytest tests/test_joint_model_and_losses.py -v
```

Expected: import fails for `avfusion.joint.model`.

- [ ] **Step 3: Implement joint model and loss helpers**

Create `avfusion/joint/model.py`:

```python
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge


class JointAVGaussianModel(nn.Module):
    def __init__(self, bridge: FTGSRendererBridge, audio_head: JointAudioHead):
        super().__init__()
        self.bridge = bridge
        self.shared_gaussians = bridge.gaussians
        self.audio_head = audio_head
        self.register_buffer("means_init", self.shared_gaussians.means.detach().clone())
        self.register_buffer("opacities_init", self.shared_gaussians.opacities.detach().clone())

    def render_rgb(self, batch: dict, sh_degree: int | None = None) -> Tensor:
        return self.bridge.render_rgb(batch, sh_degree=sh_degree)

    def render_audio(self, t: Tensor, source_audio: Tensor) -> Tensor:
        return self.audio_head(self.bridge.query_state(t), source_audio)

    def parameter_groups(self, shared_lr: float, audio_lr: float) -> list[dict]:
        return [
            {"name": "shared", "params": list(self.shared_gaussians.parameters()), "lr": shared_lr},
            {"name": "audio", "params": list(self.audio_head.parameters()), "lr": audio_lr},
        ]

    def freeze_shared(self) -> None:
        for parameter in self.shared_gaussians.parameters():
            parameter.requires_grad_(False)

    def unfreeze_shared_geometry(self) -> None:
        for name, parameter in self.shared_gaussians.named_parameters():
            parameter.requires_grad_(name in {"means", "opacities", "velocity_model"})
```

Create `avfusion/joint/losses.py`:

```python
from __future__ import annotations

import torch
from torch import Tensor, nn

from avfusion.joint.model import JointAVGaussianModel


def geometry_regularization(model: JointAVGaussianModel) -> Tensor:
    means = model.shared_gaussians.means
    opacities = model.shared_gaussians.opacities
    means_reg = nn.functional.mse_loss(means, model.means_init)
    opacity_reg = nn.functional.mse_loss(opacities.sigmoid(), model.opacities_init.sigmoid())
    return means_reg + opacity_reg


def joint_loss(
    model: JointAVGaussianModel,
    pred_rgb: Tensor,
    target_rgb: Tensor,
    pred_audio: Tensor,
    target_audio: Tensor,
    weights: dict[str, float],
) -> dict[str, Tensor]:
    zero = pred_rgb.sum() * 0
    rgb_l1 = weights.get("rgb_l1", 0.0) * nn.functional.l1_loss(pred_rgb, target_rgb.to(pred_rgb))
    audio_l1 = weights.get("audio_l1", 0.0) * nn.functional.l1_loss(pred_audio, target_audio.to(pred_audio))
    geo = weights.get("geo", 0.0) * geometry_regularization(model)
    total = zero + rgb_l1 + audio_l1 + geo
    return {"total": total, "rgb_l1": rgb_l1.detach(), "audio_l1": audio_l1.detach(), "geo": geo.detach()}
```

Modify `avfusion/joint/__init__.py`:

```python
"""Joint audio-visual Gaussian route."""

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSDependencyError, FTGSRendererBridge
from avfusion.joint.model import JointAVGaussianModel

__all__ = [
    "FTGSDependencyError",
    "FTGSRendererBridge",
    "JointAVGaussianModel",
    "JointAudioHead",
]
```

- [ ] **Step 4: Run joint model tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with torch pytest tests/test_joint_model_and_losses.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit joint model**

Run:

```bash
git add avfusion/joint/__init__.py avfusion/joint/model.py avfusion/joint/losses.py tests/test_joint_model_and_losses.py
git commit -m "feat: add joint AV model"
```

---

### Task 4: Joint Training CLI And Checkpoint Format

**Files:**
- Create: `avfusion/train/train_joint_av_gaussians.py`
- Create: `tests/test_train_joint_av_gaussians.py`

- [ ] **Step 1: Write failing training CLI tests**

Create `tests/test_train_joint_av_gaussians.py`:

```python
import torch

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.model import JointAVGaussianModel
from avfusion.train.train_joint_av_gaussians import build_arg_parser, save_joint_checkpoint
from tests.test_joint_ftgspp_bridge import FakeGaussians


def test_joint_train_parser_accepts_route_b_arguments():
    args = build_arg_parser().parse_args(
        [
            "--manifest",
            "manifest.json",
            "--ftgspp-checkpoint",
            "gaussians.pt",
            "--output",
            "joint.pt",
            "--warmup-steps",
            "2",
            "--joint-steps",
            "3",
        ]
    )

    assert args.warmup_steps == 2
    assert args.joint_steps == 3


def test_save_joint_checkpoint_records_route_and_config(tmp_path):
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4))
    output = tmp_path / "joint.pt"

    save_joint_checkpoint(
        output_path=output,
        model=model,
        ftgspp_checkpoint="gaussians.pt",
        manifest_path="manifest.json",
        config={"top_k": 4},
        stage="joint",
        loss_history=[{"step": 0, "total": 1.0}],
    )

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["route"] == "B_joint_av"
    assert checkpoint["stage"] == "joint"
    assert checkpoint["audio_head"]["mono_gain"].shape == (4, 1)
```

- [ ] **Step 2: Run training CLI tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with torch pytest tests/test_train_joint_av_gaussians.py -v
```

Expected: import fails for `avfusion.train.train_joint_av_gaussians`.

- [ ] **Step 3: Implement parser and checkpoint saving**

Create `avfusion/train/train_joint_av_gaussians.py`:

```python
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.model import JointAVGaussianModel


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Route B joint audio-visual Gaussians.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ftgspp-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--joint-steps", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=8192)
    parser.add_argument("--audio-lr", type=float, default=5e-4)
    parser.add_argument("--shared-lr", type=float, default=1e-5)
    return parser


def save_joint_checkpoint(
    output_path: str | Path,
    model: JointAVGaussianModel,
    ftgspp_checkpoint: str,
    manifest_path: str,
    config: dict[str, Any],
    stage: str,
    loss_history: list[dict[str, float]],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": "B_joint_av",
            "ftgspp_checkpoint": str(ftgspp_checkpoint),
            "manifest_path": str(manifest_path),
            "shared_gaussians": model.shared_gaussians,
            "audio_head": model.audio_head.state_dict(),
            "config": dict(config),
            "stage": stage,
            "loss_history": list(loss_history),
        },
        output_path,
    )


def build_model(ftgspp_checkpoint: str | Path, top_k: int) -> JointAVGaussianModel:
    bridge = FTGSRendererBridge.load_checkpoint(ftgspp_checkpoint)
    audio_head = JointAudioHead(num_points=len(bridge.gaussians.means), top_k=top_k)
    return JointAVGaussianModel(bridge, audio_head)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    model = build_model(args.ftgspp_checkpoint, args.top_k)
    save_joint_checkpoint(
        output_path=args.output,
        model=model,
        ftgspp_checkpoint=args.ftgspp_checkpoint,
        manifest_path=args.manifest,
        config=vars(args),
        stage="initialized",
        loss_history=[],
    )
    print({"route": "B_joint_av", "stage": "initialized", "output": args.output})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run training CLI tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with torch pytest tests/test_train_joint_av_gaussians.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit training CLI scaffold**

Run:

```bash
git add avfusion/train/train_joint_av_gaussians.py tests/test_train_joint_av_gaussians.py
git commit -m "feat: scaffold joint AV trainer"
```

---

### Task 5: Config, Scripts, README, And Smoke Commands

**Files:**
- Create: `configs/scene1_opera_b_joint_av.yaml`
- Create: `scripts/train_joint_scene1_opera.sh`
- Create: `scripts/eval_joint_scene1_opera.sh`
- Modify: `README.md`
- Test: `tests/test_project_config.py`

- [ ] **Step 1: Write failing config/script tests**

Append to `tests/test_project_config.py`:

```python
def test_scene1_route_b_config_and_scripts_are_separate_from_route_a():
    cfg_path = Path("configs/scene1_opera_b_joint_av.yaml")
    train_script = Path("scripts/train_joint_scene1_opera.sh").read_text()
    eval_script = Path("scripts/eval_joint_scene1_opera.sh").read_text()

    assert cfg_path.exists()
    text = cfg_path.read_text()
    assert "route: B_joint_av" in text
    assert "runs/scene1_opera_b_joint_av" in text
    assert "train_joint_av_gaussians" in train_script
    assert "runs/scene1_opera_b_joint_av" in train_script
    assert "eval_audio" in eval_script
    assert "runs/scene1_opera_b_joint_av" in eval_script
```

- [ ] **Step 2: Run config/script test to verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with pyyaml pytest tests/test_project_config.py::test_scene1_route_b_config_and_scripts_are_separate_from_route_a -v
```

Expected: fails because `configs/scene1_opera_b_joint_av.yaml` does not exist.

- [ ] **Step 3: Add Route B config and scripts**

Create `configs/scene1_opera_b_joint_av.yaml`:

```yaml
route: B_joint_av
scene: scene1_opera
run_dir: /mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_b_joint_av
manifest: /mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/scene_manifest.json
ftgspp_checkpoint: /mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/ftgspp/scene1_opera/00/gaussians.pt
output: /mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_b_joint_av/joint_av.pt
top_k: 8192
warmup_steps: 1000
joint_steps: 1000
audio_lr: 0.0005
shared_lr: 0.00001
```

Create `scripts/train_joint_scene1_opera.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene1_opera_b_joint_av"
mkdir -p "${RUN_DIR}"

PYTHONPATH="${ROOT}:/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus:${PYTHONPATH:-}" \
conda run -n avcloud python -m avfusion.train.train_joint_av_gaussians \
  --manifest "${ROOT}/runs/scene1_opera_a/scene_manifest.json" \
  --ftgspp-checkpoint "${ROOT}/runs/scene1_opera_a/ftgspp/scene1_opera/00/gaussians.pt" \
  --output "${RUN_DIR}/joint_av.pt" \
  --warmup-steps 1000 \
  --joint-steps 1000 \
  --top-k 8192
```

Create `scripts/eval_joint_scene1_opera.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene1_opera_b_joint_av"

PYTHONPATH="${ROOT}:${PYTHONPATH:-}" \
conda run -n avcloud python -m avfusion.eval.eval_audio \
  --manifest "${ROOT}/runs/scene1_opera_a/scene_manifest.json" \
  --checkpoint "${RUN_DIR}/joint_av.pt" \
  --output-dir "${RUN_DIR}/eval"
```

Modify `README.md` by adding:

```markdown
## Route B Joint AV Gaussian

Route B keeps Route A intact and fine-tunes a shared FTGS++ Gaussian checkpoint with a trainable audio head:

```bash
scripts/train_joint_scene1_opera.sh
scripts/eval_joint_scene1_opera.sh
```

Route B writes outputs under `runs/scene1_opera_b_joint_av/`.
```

- [ ] **Step 4: Run config/script test to verify it passes**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with pyyaml pytest tests/test_project_config.py::test_scene1_route_b_config_and_scripts_are_separate_from_route_a -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit config and scripts**

Run:

```bash
chmod +x scripts/train_joint_scene1_opera.sh scripts/eval_joint_scene1_opera.sh
git add configs/scene1_opera_b_joint_av.yaml scripts/train_joint_scene1_opera.sh scripts/eval_joint_scene1_opera.sh README.md tests/test_project_config.py
git commit -m "feat: add route B commands"
```

---

### Task 6: Verification And Remote Sync

**Files:**
- No new source files unless earlier tasks reveal a small fix.

- [ ] **Step 1: Run shell syntax check**

Run:

```bash
bash -n scripts/*.sh
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run full test suite**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with pyyaml --with numpy --with soundfile --with torch pytest -v
```

Expected: all tests pass. Existing CUDA driver warnings from the uv torch environment are acceptable if tests pass.

- [ ] **Step 3: Run CLI help in the intended FTGS++/avcloud environment**

Run:

```bash
PYTHONPATH=/mnt/sda/lisujing/Dataset/AVGaussianFusion:/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus \
conda run -n avcloud python -m avfusion.train.train_joint_av_gaussians --help
```

Expected: help text prints and includes `--ftgspp-checkpoint`, `--warmup-steps`, and `--joint-steps`.

- [ ] **Step 4: Confirm Route A remains available**

Run:

```bash
UV_CACHE_DIR=/tmp/avgaussianfusion-uv-cache uv run --with pytest --with pyyaml --with numpy --with soundfile --with torch pytest tests/test_train_frozen_carrier_audio.py tests/test_eval_audio.py -v
```

Expected: Route A training/eval tests pass.

- [ ] **Step 5: Check git status**

Run:

```bash
git status --short --branch --ignored
```

Expected: branch tracks `origin/impl/frozen-carrier-audiogs`; only ignored `.venv/` and `runs/` may remain.

- [ ] **Step 6: Push commits**

Run:

```bash
git push
```

Expected: branch updates on `git@github.com:lsj2004/AVGaussianFusion.git`.

---

## Self-Review Notes

- Spec coverage: Tasks cover the bridge to FTGS++ renderer, audio head, joint model, losses, checkpoint format, config/scripts, and verification. The first implementation produces a testable Route B scaffold with differentiable RGB/audio paths and explicit checkpointing; real long-running optimization can then reuse the same trainer entrypoint.
- Red-flag scan: This plan avoids unresolved task markers and gives concrete paths, test code, implementation code, commands, and expected outputs.
- Type consistency: `FTGSRendererBridge`, `JointAudioHead`, `JointAVGaussianModel`, and `save_joint_checkpoint` names are consistent across tasks.
