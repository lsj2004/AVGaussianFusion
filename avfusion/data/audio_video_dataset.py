from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from avfusion.data.manifest import SceneManifest


def _read_audio_crop(
    path: str,
    crop_samples: int,
    sample_rate: int,
    channels: int,
) -> torch.Tensor:
    audio, actual_sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if int(actual_sample_rate) != int(sample_rate):
        raise ValueError(
            f"sample rate mismatch for {path}: "
            f"manifest={sample_rate}, wav={actual_sample_rate}"
        )
    if audio.shape[1] != int(channels):
        raise ValueError(
            f"channel mismatch for {path}: manifest={channels}, wav={audio.shape[1]}"
        )
    if len(audio) < crop_samples:
        pad = np.zeros((crop_samples - len(audio), audio.shape[1]), dtype=np.float32)
        audio = np.concatenate([audio, pad], axis=0)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[1] != 2:
        raise ValueError(f"expected mono or stereo audio for {path}, got {audio.shape[1]}")
    audio = audio[:crop_samples, :]
    return torch.from_numpy(audio.T.copy())


def _read_audio_full(path: str, sample_rate: int, channels: int) -> torch.Tensor:
    audio, actual_sample_rate = sf.read(path, always_2d=True, dtype="float32")
    if int(actual_sample_rate) != int(sample_rate):
        raise ValueError(
            f"sample rate mismatch for {path}: "
            f"manifest={sample_rate}, wav={actual_sample_rate}"
        )
    if audio.shape[1] != int(channels):
        raise ValueError(
            f"channel mismatch for {path}: manifest={channels}, wav={audio.shape[1]}"
        )
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[1] != 2:
        raise ValueError(f"expected mono or stereo audio for {path}, got {audio.shape[1]}")
    return torch.from_numpy(audio.T.copy())


def _crop_with_padding(audio: torch.Tensor, start_sample: int, crop_samples: int) -> torch.Tensor:
    end_sample = start_sample + crop_samples
    read_start = max(start_sample, 0)
    read_end = min(end_sample, int(audio.shape[-1]))
    crop = torch.zeros(audio.shape[0], crop_samples, dtype=audio.dtype)
    if read_end <= read_start:
        return crop
    dst_start = read_start - start_sample
    dst_end = dst_start + (read_end - read_start)
    crop[:, dst_start:dst_end] = audio[:, read_start:read_end]
    return crop


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
        self._source_audio = _read_audio_crop(
            self.manifest.audio.source_path,
            self.manifest.audio.crop_samples,
            self.manifest.audio.sample_rate,
            self.manifest.audio.channels,
        )

    def __len__(self) -> int:
        return len(self.camera_names)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        camera = self.camera_names[idx]
        record = self.manifest.cameras[camera]
        crop_samples = self.manifest.audio.crop_samples
        return {
            "camera": camera,
            "source_audio": self._source_audio.clone(),
            "target_audio": _read_audio_crop(
                record.audio_path,
                crop_samples,
                self.manifest.audio.sample_rate,
                self.manifest.audio.channels,
            ),
        }


class TimedAudioCropper:
    def __init__(
        self,
        manifest_path: str | Path,
        crop_seconds: float,
        mode: str = "center",
    ):
        self.manifest = SceneManifest.load(manifest_path)
        if mode != "center":
            raise ValueError(f"only centered audio crops are supported, got {mode!r}")
        self.mode = mode
        self.crop_seconds = float(crop_seconds)
        if self.crop_seconds <= 0:
            raise ValueError(f"crop_seconds must be positive, got {self.crop_seconds}")
        self.sample_rate = int(self.manifest.audio.sample_rate)
        self.crop_samples = max(1, int(round(self.crop_seconds * self.sample_rate)))
        self._audio_cache: dict[str, torch.Tensor] = {}

    def get_crop(self, camera: str, time_seconds: float | torch.Tensor) -> dict[str, str | int | torch.Tensor]:
        camera_name = str(camera)
        if camera_name not in self.manifest.cameras:
            raise KeyError(f"unknown camera {camera_name}")
        t = float(torch.as_tensor(time_seconds).reshape(-1)[0].item())
        center_sample = int(round(t * self.sample_rate))
        start_sample = center_sample - self.crop_samples // 2
        source_audio = _crop_with_padding(
            self._load_audio(self.manifest.audio.source_path),
            start_sample,
            self.crop_samples,
        )
        target_audio = _crop_with_padding(
            self._load_audio(self.manifest.cameras[camera_name].audio_path),
            start_sample,
            self.crop_samples,
        )
        return {
            "camera": camera_name,
            "time": torch.tensor([[t]], dtype=torch.float32),
            "start_sample": int(start_sample),
            "source_audio": source_audio,
            "target_audio": target_audio,
        }

    def _load_audio(self, path: str) -> torch.Tensor:
        if path not in self._audio_cache:
            self._audio_cache[path] = _read_audio_full(
                path,
                self.manifest.audio.sample_rate,
                self.manifest.audio.channels,
            )
        return self._audio_cache[path]
