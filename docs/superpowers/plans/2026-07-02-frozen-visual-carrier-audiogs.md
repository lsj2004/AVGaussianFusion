# Frozen Visual Carrier AudioGS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Route A prototype: FreeTimeGS++ produces a frozen dynamic visual Gaussian carrier, and AVGaussianFusion trains AudioGS-style optimizable acoustic parameters on top of that carrier for `scene1_opera`.

**Architecture:** AVGaussianFusion is a small adapter project, not a forked rewrite of either baseline. It owns a unified scene manifest, a frozen carrier interface, a visual-to-acoustic adapter, a minimal AudioGS-style acoustic parameterization, audio losses, and scripts that call or consume FreeTimeGS++ / audioGS-replay artifacts.

**Tech Stack:** Python 3, PyTorch, NumPy, soundfile, PyYAML, pytest. FreeTimeGS++ integration is through local checkpoint/export files under `/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus`.

---

## File Structure

- Create `pyproject.toml`: package metadata and pytest config.
- Create `README.md`: quick project overview and first-run commands.
- Create `configs/scene1_opera_a_frozen_carrier.yaml`: source roots, split defaults, audio window defaults, output paths.
- Create `avfusion/__init__.py`: package marker and version.
- Create `avfusion/data/manifest.py`: dataclasses for manifest, camera records, and audio/video timing.
- Create `avfusion/data/build_scene_manifest.py`: CLI and library function for building `scene_manifest.json`.
- Create `avfusion/data/audio_video_dataset.py`: load 3-second source/target audio crops and pose metadata from manifest.
- Create `avfusion/visual/carrier.py`: frozen visual carrier dataclass and query interface.
- Create `avfusion/visual/export_ftgspp_carrier.py`: convert a FreeTimeGS++ checkpoint into `carrier.pt`.
- Create `avfusion/adapters/visual_to_acoustic.py`: top-k visual Gaussian selection and acoustic carrier initialization.
- Create `avfusion/audio/acoustic_gaussians.py`: trainable AudioGS-style acoustic attributes.
- Create `avfusion/audio/losses.py`: STFT and waveform shape-safe losses.
- Create `avfusion/audio/renderer.py`: minimal differentiable acoustic renderer for smoke tests.
- Create `avfusion/train/train_frozen_carrier_audio.py`: Stage 2 training loop with frozen-carrier assertions.
- Create `avfusion/eval/eval_audio.py`: render held-out WAV and JSON metric summary.
- Create `scripts/prepare_scene1_opera.sh`: build manifest.
- Create `scripts/run_stage1_ftgspp.sh`: call FreeTimeGS++ Stage 1.
- Create `scripts/export_stage1_carrier.sh`: export carrier.
- Create `scripts/train_stage2_audio.sh`: train acoustic parameters.
- Create `scripts/eval_scene1_opera.sh`: evaluate held-out audio.
- Create tests under `tests/` matching the modules above.

## Task 1: Project Scaffold And Config

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `configs/scene1_opera_a_frozen_carrier.yaml`
- Create: `avfusion/__init__.py`
- Test: `tests/test_project_config.py`

- [ ] **Step 1: Write the failing config test**

Create `tests/test_project_config.py`:

```python
from pathlib import Path

import yaml


def test_scene1_config_has_route_a_defaults():
    cfg_path = Path("configs/scene1_opera_a_frozen_carrier.yaml")
    assert cfg_path.exists()
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["scene"]["id"] == "scene1_opera"
    assert cfg["split"]["heldout_camera"] == "cam10"
    assert cfg["audio"]["source"] == "near.wav"
    assert cfg["audio"]["crop_seconds"] == 3.0
    assert cfg["audio"]["enable_phase_modeling"] is False
    assert cfg["paths"]["visual_root"] == "/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera"
    assert cfg["paths"]["audio_root"] == "/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --with pytest --with pyyaml pytest tests/test_project_config.py -v
```

Expected: FAIL because `configs/scene1_opera_a_frozen_carrier.yaml` does not exist.

- [ ] **Step 3: Add minimal project files**

Create `pyproject.toml`:

```toml
[project]
name = "avgaussianfusion"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "pyyaml",
  "soundfile",
  "torch",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Create `avfusion/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `configs/scene1_opera_a_frozen_carrier.yaml`:

```yaml
scene:
  id: scene1_opera

paths:
  visual_root: /mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera
  audio_root: /mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera
  work_dir: /mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a
  manifest: /mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/scene_manifest.json
  carrier: /mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/carrier.pt

split:
  heldout_camera: cam10

audio:
  source: near.wav
  sample_rate: 16000
  channels: 2
  crop_seconds: 3.0
  enable_phase_modeling: false

visual:
  fps: 30.0
  num_frames: 150

adapter:
  top_k: 8192

train:
  seed: 42
  max_steps: 1000
  lr: 0.0005
```

Create `README.md`:

```markdown
# AVGaussianFusion

Route A prototype for `scene1_opera`: train FreeTimeGS++ as the dynamic visual Gaussian carrier, freeze it, then optimize AudioGS-style acoustic Gaussian attributes on top of that carrier.

First config:

```bash
configs/scene1_opera_a_frozen_carrier.yaml
```
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --with pytest --with pyyaml pytest tests/test_project_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md configs/scene1_opera_a_frozen_carrier.yaml avfusion/__init__.py tests/test_project_config.py
git commit -m "chore: scaffold AVGaussianFusion config"
```

## Task 2: Unified Scene Manifest

**Files:**
- Create: `avfusion/data/__init__.py`
- Create: `avfusion/data/manifest.py`
- Create: `avfusion/data/build_scene_manifest.py`
- Create: `scripts/prepare_scene1_opera.sh`
- Test: `tests/test_manifest_builder.py`

- [ ] **Step 1: Write the failing manifest tests**

Create `tests/test_manifest_builder.py`:

```python
import json
from pathlib import Path

import pytest

from avfusion.data.build_scene_manifest import build_manifest


VISUAL_ROOT = Path("/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera")
AUDIO_ROOT = Path("/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera")


def test_build_manifest_matches_camera_sets(tmp_path):
    manifest = build_manifest(
        scene_id="scene1_opera",
        visual_root=VISUAL_ROOT,
        audio_root=AUDIO_ROOT,
        heldout_camera="cam10",
        fps=30.0,
        num_frames=150,
        sample_rate=16000,
        crop_seconds=3.0,
    )

    assert manifest.scene_id == "scene1_opera"
    assert len(manifest.cameras) == 39
    assert manifest.train_cameras[0] == "cam00"
    assert "cam10" not in manifest.train_cameras
    assert manifest.eval_cameras == ["cam10"]
    assert manifest.frame_times[0] == 0.0
    assert manifest.frame_times[-1] == pytest.approx(149 / 30.0)
    assert manifest.audio.sample_rate == 16000
    assert manifest.audio.crop_samples == 48000


def test_manifest_writes_json(tmp_path):
    out = tmp_path / "scene_manifest.json"
    manifest = build_manifest(
        scene_id="scene1_opera",
        visual_root=VISUAL_ROOT,
        audio_root=AUDIO_ROOT,
        heldout_camera="cam10",
        output_path=out,
    )

    loaded = json.loads(out.read_text())
    assert loaded["scene_id"] == manifest.scene_id
    assert loaded["eval_cameras"] == ["cam10"]
    assert loaded["cameras"]["cam10"]["video_path"].endswith("cam10.mp4")
    assert loaded["cameras"]["cam10"]["audio_path"].endswith("cam10.wav")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --with pytest --with pyyaml pytest tests/test_manifest_builder.py -v
```

Expected: FAIL because `avfusion.data.build_scene_manifest` is missing.

- [ ] **Step 3: Implement manifest dataclasses**

Create `avfusion/data/__init__.py`:

```python
"""Data loading and manifest helpers."""
```

Create `avfusion/data/manifest.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraRecord:
    name: str
    index: int
    video_path: str
    audio_path: str


@dataclass(frozen=True)
class AudioSpec:
    sample_rate: int
    channels: int
    crop_seconds: float
    crop_samples: int
    source_path: str


@dataclass(frozen=True)
class SceneManifest:
    scene_id: str
    visual_root: str
    audio_root: str
    fps: float
    num_frames: int
    frame_times: list[float]
    cameras: dict[str, CameraRecord]
    train_cameras: list[str]
    eval_cameras: list[str]
    audio: AudioSpec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneManifest":
        cameras = {
            name: CameraRecord(**record)
            for name, record in data["cameras"].items()
        }
        audio = AudioSpec(**data["audio"])
        return cls(
            scene_id=data["scene_id"],
            visual_root=data["visual_root"],
            audio_root=data["audio_root"],
            fps=float(data["fps"]),
            num_frames=int(data["num_frames"]),
            frame_times=[float(x) for x in data["frame_times"]],
            cameras=cameras,
            train_cameras=list(data["train_cameras"]),
            eval_cameras=list(data["eval_cameras"]),
            audio=audio,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SceneManifest":
        import json

        return cls.from_dict(json.loads(Path(path).read_text()))
```

- [ ] **Step 4: Implement manifest builder**

Create `avfusion/data/build_scene_manifest.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avfusion.data.manifest import AudioSpec, CameraRecord, SceneManifest


def _camera_name_from_path(path: Path) -> str:
    return path.stem


def _sorted_camera_names(root: Path, suffix: str) -> list[str]:
    return sorted(_camera_name_from_path(p) for p in root.glob(f"cam*{suffix}"))


def build_manifest(
    scene_id: str,
    visual_root: str | Path,
    audio_root: str | Path,
    heldout_camera: str = "cam10",
    fps: float = 30.0,
    num_frames: int = 150,
    sample_rate: int = 16000,
    crop_seconds: float = 3.0,
    output_path: str | Path | None = None,
) -> SceneManifest:
    visual_root = Path(visual_root)
    audio_root = Path(audio_root)
    aligned_audio_root = audio_root / "aligned_16k_stereo"

    video_names = _sorted_camera_names(visual_root, ".mp4")
    audio_names = _sorted_camera_names(aligned_audio_root, ".wav")
    if video_names != audio_names:
        missing_audio = sorted(set(video_names) - set(audio_names))
        missing_video = sorted(set(audio_names) - set(video_names))
        raise ValueError(
            f"camera mismatch: missing_audio={missing_audio}, missing_video={missing_video}"
        )
    if heldout_camera not in video_names:
        raise ValueError(f"heldout camera {heldout_camera!r} is not present")

    source_path = aligned_audio_root / "near.wav"
    if not source_path.exists():
        raise FileNotFoundError(f"missing source audio: {source_path}")

    cameras = {
        name: CameraRecord(
            name=name,
            index=int(name.replace("cam", "")),
            video_path=str(visual_root / f"{name}.mp4"),
            audio_path=str(aligned_audio_root / f"{name}.wav"),
        )
        for name in video_names
    }
    train_cameras = [name for name in video_names if name != heldout_camera]
    frame_times = [idx / fps for idx in range(num_frames)]
    manifest = SceneManifest(
        scene_id=scene_id,
        visual_root=str(visual_root),
        audio_root=str(audio_root),
        fps=float(fps),
        num_frames=int(num_frames),
        frame_times=frame_times,
        cameras=cameras,
        train_cameras=train_cameras,
        eval_cameras=[heldout_camera],
        audio=AudioSpec(
            sample_rate=int(sample_rate),
            channels=2,
            crop_seconds=float(crop_seconds),
            crop_samples=int(round(sample_rate * crop_seconds)),
            source_path=str(source_path),
        ),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest.to_dict(), indent=2))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default="scene1_opera")
    parser.add_argument("--visual-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--heldout-camera", default="cam10")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_manifest(
        scene_id=args.scene_id,
        visual_root=args.visual_root,
        audio_root=args.audio_root,
        heldout_camera=args.heldout_camera,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add prepare script**

Create `scripts/prepare_scene1_opera.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene1_opera_a"
mkdir -p "${RUN_DIR}"

python -m avfusion.data.build_scene_manifest \
  --scene-id scene1_opera \
  --visual-root /mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera \
  --audio-root /mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera \
  --heldout-camera cam10 \
  --output "${RUN_DIR}/scene_manifest.json"
```

Run:

```bash
chmod +x scripts/prepare_scene1_opera.sh
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
uv run --with pytest --with pyyaml pytest tests/test_manifest_builder.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add avfusion/data tests/test_manifest_builder.py scripts/prepare_scene1_opera.sh
git commit -m "feat: build unified scene manifest"
```

## Task 3: Audio/Video Dataset Crops

**Files:**
- Create: `avfusion/data/audio_video_dataset.py`
- Test: `tests/test_audio_video_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Create `tests/test_audio_video_dataset.py`:

```python
from pathlib import Path

import torch

from avfusion.data.audio_video_dataset import AudioCropDataset
from avfusion.data.build_scene_manifest import build_manifest


VISUAL_ROOT = Path("/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera")
AUDIO_ROOT = Path("/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera")


def test_audio_crop_dataset_returns_source_and_target(tmp_path):
    manifest_path = tmp_path / "scene_manifest.json"
    build_manifest(
        scene_id="scene1_opera",
        visual_root=VISUAL_ROOT,
        audio_root=AUDIO_ROOT,
        heldout_camera="cam10",
        output_path=manifest_path,
    )

    dataset = AudioCropDataset(manifest_path, split="train")
    sample = dataset[0]

    assert sample["camera"] == "cam00"
    assert sample["source_audio"].shape == (2, 48000)
    assert sample["target_audio"].shape == (2, 48000)
    assert sample["source_audio"].dtype == torch.float32
    assert sample["target_audio"].dtype == torch.float32


def test_audio_crop_dataset_eval_uses_heldout_camera(tmp_path):
    manifest_path = tmp_path / "scene_manifest.json"
    build_manifest(
        scene_id="scene1_opera",
        visual_root=VISUAL_ROOT,
        audio_root=AUDIO_ROOT,
        heldout_camera="cam10",
        output_path=manifest_path,
    )

    dataset = AudioCropDataset(manifest_path, split="eval")
    assert len(dataset) == 1
    assert dataset[0]["camera"] == "cam10"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --with pytest --with numpy --with soundfile --with torch pytest tests/test_audio_video_dataset.py -v
```

Expected: FAIL because `AudioCropDataset` is missing.

- [ ] **Step 3: Implement crop dataset**

Create `avfusion/data/audio_video_dataset.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from avfusion.data.manifest import SceneManifest


def _read_audio_crop(path: str, crop_samples: int) -> torch.Tensor:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if len(audio) < crop_samples:
        pad = np.zeros((crop_samples - len(audio), audio.shape[1]), dtype=np.float32)
        audio = np.concatenate([audio, pad], axis=0)
    audio = audio[:crop_samples, :2]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return torch.from_numpy(audio.T.copy())


class AudioCropDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str):
        self.manifest = SceneManifest.load(manifest_path)
        if split not in {"train", "eval"}:
            raise ValueError(f"split must be train or eval, got {split!r}")
        self.split = split
        self.camera_names = (
            self.manifest.train_cameras if split == "train" else self.manifest.eval_cameras
        )

    def __len__(self) -> int:
        return len(self.camera_names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        camera = self.camera_names[idx]
        record = self.manifest.cameras[camera]
        crop_samples = self.manifest.audio.crop_samples
        return {
            "camera": camera,
            "source_audio": _read_audio_crop(self.manifest.audio.source_path, crop_samples),
            "target_audio": _read_audio_crop(record.audio_path, crop_samples),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --with pytest --with numpy --with soundfile --with torch pytest tests/test_audio_video_dataset.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add avfusion/data/audio_video_dataset.py tests/test_audio_video_dataset.py
git commit -m "feat: load synchronized audio crops"
```

## Task 4: Frozen Visual Carrier Interface

**Files:**
- Create: `avfusion/visual/__init__.py`
- Create: `avfusion/visual/carrier.py`
- Test: `tests/test_visual_carrier.py`

- [ ] **Step 1: Write failing carrier tests**

Create `tests/test_visual_carrier.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_visual_carrier.py -v
```

Expected: FAIL because `FrozenVisualCarrier` is missing.

- [ ] **Step 3: Implement carrier interface**

Create `avfusion/visual/__init__.py`:

```python
"""Frozen visual Gaussian carrier interfaces."""
```

Create `avfusion/visual/carrier.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CarrierState:
    xyz: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacity: torch.Tensor


class FrozenVisualCarrier:
    def __init__(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        times: torch.Tensor,
        durations: torch.Tensor,
        velocities: torch.Tensor,
        max_duration: float,
    ):
        self.means = means.detach().float()
        self.scales = scales.detach().float()
        self.quats = quats.detach().float()
        self.opacities = opacities.detach().float()
        self.times = times.detach().float()
        self.durations = durations.detach().float()
        self.velocities = velocities.detach().float()
        self.max_duration = float(max_duration)
        self.assert_frozen()

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def query(self, t: float | torch.Tensor) -> CarrierState:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), dtype=self.means.dtype, device=self.means.device)
        t = t.reshape(1, 1).to(self.means)
        xyz = self.means + (t - self.times) * self.velocities
        if self.max_duration == float("inf"):
            tscale = torch.exp(self.durations)
        else:
            tscale = self.max_duration / 6.0 * torch.sigmoid(self.durations)
        temporal_opacity = torch.exp(-0.5 * ((t - self.times) / tscale) ** 2)
        opacity = torch.sigmoid(self.opacities) * temporal_opacity
        return CarrierState(
            xyz=xyz.detach(),
            scales=self.scales.detach(),
            quats=self.quats.detach(),
            opacity=opacity.detach(),
        )

    def assert_frozen(self) -> None:
        for name in ("means", "scales", "quats", "opacities", "times", "durations", "velocities"):
            tensor = getattr(self, name)
            if tensor.requires_grad:
                raise RuntimeError(f"carrier tensor {name} must be frozen")

    def save(self, path: str) -> None:
        torch.save(
            {
                "means": self.means,
                "scales": self.scales,
                "quats": self.quats,
                "opacities": self.opacities,
                "times": self.times,
                "durations": self.durations,
                "velocities": self.velocities,
                "max_duration": self.max_duration,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FrozenVisualCarrier":
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_visual_carrier.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add avfusion/visual tests/test_visual_carrier.py
git commit -m "feat: add frozen visual carrier interface"
```

## Task 5: Visual-To-Acoustic Adapter

**Files:**
- Create: `avfusion/adapters/__init__.py`
- Create: `avfusion/adapters/visual_to_acoustic.py`
- Test: `tests/test_visual_to_acoustic_adapter.py`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/test_visual_to_acoustic_adapter.py`:

```python
import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier, select_topk_acoustic_carrier
from avfusion.visual.carrier import FrozenVisualCarrier


def test_select_topk_acoustic_carrier_orders_by_opacity():
    carrier = FrozenVisualCarrier(
        means=torch.arange(15, dtype=torch.float32).reshape(5, 3),
        scales=torch.zeros(5, 3),
        quats=torch.ones(5, 4),
        opacities=torch.tensor([[-2.0], [3.0], [0.0], [1.0], [2.0]]),
        times=torch.zeros(5, 1),
        durations=torch.zeros(5, 1),
        velocities=torch.zeros(5, 3),
        max_duration=float("inf"),
    )

    acoustic = select_topk_acoustic_carrier(carrier, t=0.0, top_k=3)

    assert isinstance(acoustic, AcousticCarrier)
    assert acoustic.xyz.shape == (3, 3)
    assert acoustic.visual_indices.tolist() == [1, 4, 3]
    assert not acoustic.xyz.requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_visual_to_acoustic_adapter.py -v
```

Expected: FAIL because adapter module is missing.

- [ ] **Step 3: Implement adapter**

Create `avfusion/adapters/__init__.py`:

```python
"""Adapters between visual and acoustic Gaussian representations."""
```

Create `avfusion/adapters/visual_to_acoustic.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch

from avfusion.visual.carrier import FrozenVisualCarrier


@dataclass(frozen=True)
class AcousticCarrier:
    xyz: torch.Tensor
    opacity: torch.Tensor
    visual_indices: torch.Tensor


def select_topk_acoustic_carrier(
    carrier: FrozenVisualCarrier,
    t: float,
    top_k: int,
) -> AcousticCarrier:
    state = carrier.query(t)
    k = min(int(top_k), len(carrier))
    scores = state.opacity.squeeze(-1)
    indices = torch.topk(scores, k=k, largest=True, sorted=True).indices
    return AcousticCarrier(
        xyz=state.xyz[indices].detach(),
        opacity=state.opacity[indices].detach(),
        visual_indices=indices.detach().cpu(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_visual_to_acoustic_adapter.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add avfusion/adapters tests/test_visual_to_acoustic_adapter.py
git commit -m "feat: initialize acoustic carrier from visual gaussians"
```

## Task 6: Acoustic Parameters, Renderer, And Loss

**Files:**
- Create: `avfusion/audio/__init__.py`
- Create: `avfusion/audio/acoustic_gaussians.py`
- Create: `avfusion/audio/renderer.py`
- Create: `avfusion/audio/losses.py`
- Test: `tests/test_audio_renderer_and_losses.py`

- [ ] **Step 1: Write failing audio renderer/loss tests**

Create `tests/test_audio_renderer_and_losses.py`:

```python
import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio.acoustic_gaussians import AcousticGaussianParameters
from avfusion.audio.losses import stft_magnitude_loss
from avfusion.audio.renderer import render_audio


def test_renderer_outputs_stereo_waveform_and_optimizable_params():
    carrier = AcousticCarrier(
        xyz=torch.zeros(4, 3),
        opacity=torch.ones(4, 1),
        visual_indices=torch.arange(4),
    )
    params = AcousticGaussianParameters(num_points=4)
    source = torch.randn(2, 1024)

    pred = render_audio(carrier, params, source)

    assert pred.shape == source.shape
    assert pred.requires_grad
    assert params.mono_gain.requires_grad
    assert params.diff_gain.requires_grad


def test_stft_magnitude_loss_is_scalar_and_differentiable():
    pred = torch.randn(2, 2048, requires_grad=True)
    target = torch.randn(2, 2048)

    loss = stft_magnitude_loss(pred, target, n_fft=256, hop_length=64)

    assert loss.ndim == 0
    loss.backward()
    assert pred.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_audio_renderer_and_losses.py -v
```

Expected: FAIL because audio modules are missing.

- [ ] **Step 3: Implement minimal acoustic parameters**

Create `avfusion/audio/__init__.py`:

```python
"""AudioGS-style acoustic rendering modules."""
```

Create `avfusion/audio/acoustic_gaussians.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class AcousticGaussianParameters(nn.Module):
    def __init__(self, num_points: int):
        super().__init__()
        self.mono_gain = nn.Parameter(torch.zeros(num_points, 1))
        self.diff_gain = nn.Parameter(torch.zeros(num_points, 1))

    def aggregate(self, opacity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = opacity / opacity.sum().clamp_min(1e-6)
        mono = (weights * self.mono_gain).sum().tanh()
        diff = (weights * self.diff_gain).sum().tanh()
        return mono, diff
```

- [ ] **Step 4: Implement minimal differentiable renderer**

Create `avfusion/audio/renderer.py`:

```python
from __future__ import annotations

import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio.acoustic_gaussians import AcousticGaussianParameters


def render_audio(
    carrier: AcousticCarrier,
    params: AcousticGaussianParameters,
    source_audio: torch.Tensor,
) -> torch.Tensor:
    mono, diff = params.aggregate(carrier.opacity.to(source_audio.device))
    source_audio = source_audio.to(params.mono_gain.device)
    mono_source = source_audio.mean(dim=0, keepdim=True)
    left = mono_source * (1.0 + mono + diff)
    right = mono_source * (1.0 + mono - diff)
    return torch.cat([left, right], dim=0)
```

- [ ] **Step 5: Implement STFT loss**

Create `avfusion/audio/losses.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def stft_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
    pred_spec = torch.stft(
        pred,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    target_spec = torch.stft(
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    return F.l1_loss(torch.log1p(pred_spec), torch.log1p(target_spec))
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_audio_renderer_and_losses.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add avfusion/audio tests/test_audio_renderer_and_losses.py
git commit -m "feat: add minimal acoustic renderer and loss"
```

## Task 7: Stage 2 Training Smoke Loop

**Files:**
- Create: `avfusion/train/__init__.py`
- Create: `avfusion/train/train_frozen_carrier_audio.py`
- Create: `scripts/train_stage2_audio.sh`
- Test: `tests/test_train_frozen_carrier_audio.py`

- [ ] **Step 1: Write failing training smoke test**

Create `tests/test_train_frozen_carrier_audio.py`:

```python
import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.train.train_frozen_carrier_audio import train_one_step
from avfusion.visual.carrier import FrozenVisualCarrier


def test_train_one_step_updates_only_acoustic_params():
    carrier = FrozenVisualCarrier(
        means=torch.zeros(8, 3),
        scales=torch.zeros(8, 3),
        quats=torch.ones(8, 4),
        opacities=torch.ones(8, 1),
        times=torch.zeros(8, 1),
        durations=torch.zeros(8, 1),
        velocities=torch.zeros(8, 3),
        max_duration=float("inf"),
    )
    acoustic = AcousticCarrier(
        xyz=torch.zeros(4, 3),
        opacity=torch.ones(4, 1),
        visual_indices=torch.arange(4),
    )
    source = torch.randn(2, 2048)
    target = source * 0.5

    before = carrier.means.clone()
    loss_value = train_one_step(acoustic, source, target, lr=1e-2)

    assert loss_value > 0
    assert torch.allclose(carrier.means, before)
    carrier.assert_frozen()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_train_frozen_carrier_audio.py -v
```

Expected: FAIL because training module is missing.

- [ ] **Step 3: Implement one-step trainer**

Create `avfusion/train/__init__.py`:

```python
"""Training entrypoints."""
```

Create `avfusion/train/train_frozen_carrier_audio.py`:

```python
from __future__ import annotations

import argparse

import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio.acoustic_gaussians import AcousticGaussianParameters
from avfusion.audio.losses import stft_magnitude_loss
from avfusion.audio.renderer import render_audio


def train_one_step(
    acoustic_carrier: AcousticCarrier,
    source_audio: torch.Tensor,
    target_audio: torch.Tensor,
    lr: float,
) -> float:
    params = AcousticGaussianParameters(num_points=acoustic_carrier.xyz.shape[0])
    optimizer = torch.optim.Adam(params.parameters(), lr=lr)
    pred = render_audio(acoustic_carrier, params, source_audio)
    loss = stft_magnitude_loss(pred, target_audio)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    raise SystemExit(
        "CLI wiring is added after carrier export is available. "
        f"Received manifest={args.manifest} carrier={args.carrier} steps={args.steps} lr={args.lr}"
    )


if __name__ == "__main__":
    main()
```

Create `scripts/train_stage2_audio.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene1_opera_a"

python -m avfusion.train.train_frozen_carrier_audio \
  --manifest "${RUN_DIR}/scene_manifest.json" \
  --carrier "${RUN_DIR}/carrier.pt" \
  --steps 1000 \
  --lr 0.0005
```

Run:

```bash
chmod +x scripts/train_stage2_audio.sh
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_train_frozen_carrier_audio.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add avfusion/train tests/test_train_frozen_carrier_audio.py scripts/train_stage2_audio.sh
git commit -m "feat: add frozen-carrier training smoke loop"
```

## Task 8: FreeTimeGS++ Carrier Export

**Files:**
- Create: `avfusion/visual/export_ftgspp_carrier.py`
- Create: `scripts/export_stage1_carrier.sh`
- Test: `tests/test_export_ftgspp_carrier.py`

- [ ] **Step 1: Write failing export test with fake FTGS++ object**

Create `tests/test_export_ftgspp_carrier.py`:

```python
from types import SimpleNamespace

import torch

from avfusion.visual.export_ftgspp_carrier import carrier_from_ftgspp_object


def test_carrier_from_ftgspp_object_extracts_explicit_velocity():
    gs = SimpleNamespace(
        means=torch.zeros(2, 3, requires_grad=True),
        scales=torch.zeros(2, 3, requires_grad=True),
        quats=torch.ones(2, 4, requires_grad=True),
        opacities=torch.zeros(2, 1, requires_grad=True),
        times=torch.zeros(2, 1, requires_grad=True),
        durations=torch.zeros(2, 1, requires_grad=True),
        velocity_model=torch.ones(2, 3, requires_grad=True),
        max_duration=float("inf"),
    )

    carrier = carrier_from_ftgspp_object(gs)

    assert len(carrier) == 2
    assert not carrier.means.requires_grad
    assert torch.allclose(carrier.velocities, torch.ones(2, 3))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_export_ftgspp_carrier.py -v
```

Expected: FAIL because export module is missing.

- [ ] **Step 3: Implement checkpoint/object exporter**

Create `avfusion/visual/export_ftgspp_carrier.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from avfusion.visual.carrier import FrozenVisualCarrier


def _extract_velocity(gs) -> torch.Tensor:
    velocity_model = getattr(gs, "velocity_model")
    if isinstance(velocity_model, torch.Tensor):
        return velocity_model.detach()
    if hasattr(velocity_model, "detach"):
        return velocity_model.detach()
    raise TypeError(
        "Only explicit velocity tensors are supported in the first carrier exporter. "
        "VelocityField export should be added after inspecting the trained checkpoint."
    )


def carrier_from_ftgspp_object(gs) -> FrozenVisualCarrier:
    return FrozenVisualCarrier(
        means=gs.means,
        scales=gs.scales,
        quats=gs.quats,
        opacities=gs.opacities,
        times=gs.times,
        durations=gs.durations,
        velocities=_extract_velocity(gs),
        max_duration=getattr(gs, "max_duration", float("inf")),
    )


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path) -> FrozenVisualCarrier:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    gs = checkpoint.get("gaussians", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    carrier = carrier_from_ftgspp_object(gs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    carrier.save(str(output_path))
    return carrier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
```

Create `scripts/export_stage1_carrier.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene1_opera_a"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/ftgspp_checkpoint.pt" >&2
  exit 2
fi

python -m avfusion.visual.export_ftgspp_carrier \
  --checkpoint "$1" \
  --output "${RUN_DIR}/carrier.pt"
```

Run:

```bash
chmod +x scripts/export_stage1_carrier.sh
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_export_ftgspp_carrier.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add avfusion/visual/export_ftgspp_carrier.py tests/test_export_ftgspp_carrier.py scripts/export_stage1_carrier.sh
git commit -m "feat: export frozen carrier from FTGS++"
```

## Task 9: Evaluation CLI

**Files:**
- Create: `avfusion/eval/__init__.py`
- Create: `avfusion/eval/eval_audio.py`
- Create: `scripts/eval_scene1_opera.sh`
- Test: `tests/test_eval_audio.py`

- [ ] **Step 1: Write failing evaluation test**

Create `tests/test_eval_audio.py`:

```python
import json

import torch

from avfusion.eval.eval_audio import write_eval_summary


def test_write_eval_summary_outputs_json(tmp_path):
    pred = torch.zeros(2, 16)
    target = torch.ones(2, 16)
    out = tmp_path / "summary.json"

    summary = write_eval_summary(out, camera="cam10", pred=pred, target=target)

    loaded = json.loads(out.read_text())
    assert loaded["camera"] == "cam10"
    assert loaded["l1_waveform"] == summary["l1_waveform"]
    assert loaded["l1_waveform"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_eval_audio.py -v
```

Expected: FAIL because eval module is missing.

- [ ] **Step 3: Implement evaluation summary**

Create `avfusion/eval/__init__.py`:

```python
"""Evaluation helpers."""
```

Create `avfusion/eval/eval_audio.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def write_eval_summary(
    output_path: str | Path,
    camera: str,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | str]:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    summary = {
        "camera": camera,
        "l1_waveform": float(torch.mean(torch.abs(pred - target)).detach().cpu()),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    raise SystemExit(
        "Full eval CLI is wired after persistent Stage 2 checkpoints are implemented. "
        f"Received manifest={args.manifest} checkpoint={args.checkpoint} output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
```

Create `scripts/eval_scene1_opera.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene1_opera_a"

python -m avfusion.eval.eval_audio \
  --manifest "${RUN_DIR}/scene_manifest.json" \
  --checkpoint "${RUN_DIR}/stage2_audio.pt" \
  --output-dir "${RUN_DIR}/eval"
```

Run:

```bash
chmod +x scripts/eval_scene1_opera.sh
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run --with pytest --with torch pytest tests/test_eval_audio.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add avfusion/eval tests/test_eval_audio.py scripts/eval_scene1_opera.sh
git commit -m "feat: add audio evaluation summary"
```

## Task 10: Stage 1 FreeTimeGS++ Runner And Full Test Sweep

**Files:**
- Create: `scripts/run_stage1_ftgspp.sh`
- Modify: `README.md`
- Test: all tests

- [ ] **Step 1: Add Stage 1 runner script**

Create `scripts/run_stage1_ftgspp.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

FTGSPP="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
OUT="/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/ftgspp"

cd "${FTGSPP}"
uv run --no-sync ./run dynerf configs/dynerf/ftgspp "${OUT}" \
  --scenes scene1_opera \
  --from extract \
  --to render
```

Run:

```bash
chmod +x scripts/run_stage1_ftgspp.sh
```

- [ ] **Step 2: Update README with runnable order**

Modify `README.md` to contain:

```markdown
# AVGaussianFusion

Route A prototype for `scene1_opera`: train FreeTimeGS++ as the dynamic visual Gaussian carrier, freeze it, then optimize AudioGS-style acoustic Gaussian attributes on top of that carrier.

## First Route A Commands

```bash
scripts/prepare_scene1_opera.sh
scripts/run_stage1_ftgspp.sh
scripts/export_stage1_carrier.sh /path/to/ftgspp_checkpoint.pt
scripts/train_stage2_audio.sh
scripts/eval_scene1_opera.sh
```

The first implementation defaults are `cam10` held out, `near.wav` source audio, 3-second synchronized crops, and phase modeling disabled for the initial smoke test.
```

- [ ] **Step 3: Run full unit test sweep**

Run:

```bash
uv run --with pytest --with pyyaml --with numpy --with soundfile --with torch pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Generate manifest smoke artifact**

Run:

```bash
uv run --with pyyaml --with numpy --with soundfile --with torch scripts/prepare_scene1_opera.sh
```

Expected: `/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/scene_manifest.json` exists and lists 39 cameras with `cam10` in `eval_cameras`.

- [ ] **Step 5: Commit final scaffolding**

```bash
git add README.md scripts/run_stage1_ftgspp.sh
git commit -m "docs: add route A execution commands"
```

## Plan Self-Review

- Spec coverage: The plan covers the unified manifest, frozen carrier interface, visual-to-acoustic adapter, acoustic trainable parameters, audio loss, Stage 2 smoke training, carrier export, eval summaries, script entrypoints, and the Route C/B-compatible separation of modules.
- Intentional first-version limitation: the renderer in Task 6 is a smoke-testable differentiable acoustic renderer, not the full AudioGS renderer. The full AudioGS math should replace or extend `avfusion/audio/renderer.py` after the carrier/export/data path is verified.
- Default decisions: `cam10`, `near.wav`, 3-second crops, and phase disabled are encoded in config and tests.
