from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from avfusion.data.manifest import SceneManifest


def _read_audio_crop(path: str, crop_samples: int) -> torch.Tensor:
    audio, _ = sf.read(path, always_2d=True, dtype="float32")
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
            self.manifest.train_cameras
            if split == "train"
            else self.manifest.eval_cameras
        )

    def __len__(self) -> int:
        return len(self.camera_names)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        camera = self.camera_names[idx]
        record = self.manifest.cameras[camera]
        crop_samples = self.manifest.audio.crop_samples
        return {
            "camera": camera,
            "source_audio": _read_audio_crop(
                self.manifest.audio.source_path, crop_samples
            ),
            "target_audio": _read_audio_crop(record.audio_path, crop_samples),
        }
