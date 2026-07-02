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
