from __future__ import annotations

from pathlib import Path
from typing import Callable

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from avfusion.data.manifest import SceneManifest


FrameReader = Callable[[str, int], Tensor]


def _read_video_frame(path: str, frame_idx: int) -> Tensor:
    try:
        from torchvision.io import VideoReader
    except ImportError as error:
        raise ImportError(
            "Visual joint training requires torchvision video IO, or pass a custom frame_reader."
        ) from error
    reader = VideoReader(path, "video")
    metadata = reader.get_metadata().get("video", {})
    fps_values = metadata.get("fps", [30.0])
    fps = float(fps_values[0] if isinstance(fps_values, list) else fps_values)
    reader.seek(frame_idx / fps)
    frame = next(reader)["data"].permute(1, 2, 0).float() / 255.0
    return frame


def _resize_rgb(rgb: Tensor, scale: float) -> Tensor:
    if scale == 1:
        return rgb
    chw = rgb.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(chw, scale_factor=scale, mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).contiguous()


class VisualFrameDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        scale: float = 0.5,
        memmap_root: str | Path | None = None,
        frame_reader: FrameReader | None = None,
    ):
        self.manifest = SceneManifest.load(manifest_path)
        if split not in {"train", "eval"}:
            raise ValueError(f"split must be train or eval, got {split!r}")
        self.camera_names = (
            self.manifest.train_cameras
            if split == "train"
            else self.manifest.eval_cameras
        )
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        self.frame_reader = frame_reader or _read_video_frame
        self.memmap_root = Path(memmap_root) if memmap_root is not None else None
        self._memmap = self._load_memmap() if self.memmap_root is not None else None
        self._memmap_resize_factor = 1.0
        self._camera_metadata = self._load_camera_metadata()

    def _load_memmap(self) -> dict[str, object]:
        assert self.memmap_root is not None
        meta_path = self.memmap_root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing FTGS++ memmap metadata: {meta_path}")
        meta = json.loads(meta_path.read_text())
        arrays = {}
        dtype_map = {
            "torch.uint8": np.uint8,
            "torch.float32": np.float32,
        }
        for key in ("rgb", "w2c", "intrinsic", "time"):
            spec = meta[key]
            arrays[key] = np.memmap(
                self.memmap_root / f"{key}.memmap",
                mode="r",
                dtype=dtype_map[spec["dtype"]],
                shape=tuple(spec["shape"]),
            )
        return arrays

    def _load_camera_metadata(self) -> dict[str, tuple[Tensor, Tensor]]:
        camera_path = Path(self.manifest.visual_root) / "cameras.npz"
        if not camera_path.exists():
            raise FileNotFoundError(f"missing camera metadata: {camera_path}")
        if self._memmap is not None:
            original = np.load(camera_path)
            original_names = [str(name) for name in original["names"]]
            metadata = {}
            intrinsic_memmap = self._memmap["intrinsic"]
            self._memmap_resize_factor = self._requested_to_memmap_scale(original, intrinsic_memmap)
            for name in self.camera_names:
                cam_idx = int(name.replace("cam", ""))
                if name not in original_names:
                    raise KeyError(f"missing camera metadata for {name}")
                w2c = torch.tensor(self._memmap["w2c"][0, cam_idx], dtype=torch.float32)
                intrinsic = torch.tensor(
                    intrinsic_memmap[0, cam_idx],
                    dtype=torch.float32,
                )
                intrinsic = intrinsic * self._memmap_resize_factor
                intrinsic[2, 2] = 1.0
                metadata[name] = (w2c, intrinsic)
            return metadata

        data = np.load(camera_path)
        names = [str(name) for name in data["names"]]
        intrinsics = data["intrinsics"]
        w2cs = data["w2c"]
        metadata: dict[str, tuple[Tensor, Tensor]] = {}
        for idx, name in enumerate(names):
            intrinsic = torch.tensor(intrinsics[idx], dtype=torch.float32) * self.scale
            intrinsic[2, 2] = 1.0
            metadata[name] = (
                torch.tensor(w2cs[idx], dtype=torch.float32),
                intrinsic,
            )
        return metadata

    def _requested_to_memmap_scale(self, original: np.lib.npyio.NpzFile, intrinsic_memmap) -> float:
        names = [str(name) for name in original["names"]]
        first_name = self.camera_names[0]
        cam_idx = int(first_name.replace("cam", ""))
        original_idx = names.index(first_name)
        original_fx = float(original["intrinsics"][original_idx, 0, 0])
        memmap_fx = float(intrinsic_memmap[0, cam_idx, 0, 0])
        memmap_scale = memmap_fx / original_fx
        return self.scale / memmap_scale

    def __len__(self) -> int:
        return len(self.camera_names) * int(self.manifest.num_frames)

    def __getitem__(self, idx: int) -> dict[str, str | int | Tensor]:
        camera_name = self.camera_names[idx % len(self.camera_names)]
        frame_idx = (idx // len(self.camera_names)) % int(self.manifest.num_frames)
        record = self.manifest.cameras[camera_name]
        if camera_name not in self._camera_metadata:
            raise KeyError(f"missing camera metadata for {camera_name}")
        w2c, intrinsic = self._camera_metadata[camera_name]
        if self._memmap is not None:
            cam_idx = int(camera_name.replace("cam", ""))
            rgb = torch.tensor(
                np.array(self._memmap["rgb"][frame_idx, cam_idx]),
                dtype=torch.float32,
            ) / 255.0
            rgb = _resize_rgb(rgb, self._memmap_resize_factor)
            time = torch.tensor(
                np.array(self._memmap["time"][frame_idx, cam_idx]).reshape(1, 1),
                dtype=torch.float32,
            )
        else:
            rgb = _resize_rgb(self.frame_reader(record.video_path, frame_idx), self.scale)
            time = torch.tensor([[self.manifest.frame_times[frame_idx]]], dtype=torch.float32)
        return {
            "camera": camera_name,
            "frame": int(frame_idx),
            "time": time,
            "w2c": w2c.unsqueeze(0),
            "intrinsic": intrinsic.unsqueeze(0),
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "target_rgb": rgb.unsqueeze(0),
        }
