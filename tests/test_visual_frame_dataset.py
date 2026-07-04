from pathlib import Path

import numpy as np
import pytest
import torch

from avfusion.data.visual_frame_dataset import VisualFrameDataset
from tests.test_audio_video_dataset import _write_manifest


def test_visual_frame_dataset_returns_render_batch_and_scaled_rgb(tmp_path):
    manifest = _write_manifest(tmp_path)
    visual_root = tmp_path / "visual"
    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3) * 4, np.eye(3) * 5]),
        w2c=np.stack([np.eye(4), np.eye(4) * 2]),
    )

    def fake_frame_reader(path: str, frame_idx: int):
        assert Path(path).name == "cam00.mp4"
        assert frame_idx == 0
        return torch.ones(4, 6, 3)

    dataset = VisualFrameDataset(
        manifest,
        split="train",
        scale=0.5,
        frame_reader=fake_frame_reader,
    )

    sample = dataset[0]

    assert sample["camera"] == "cam00"
    assert sample["frame"] == 0
    assert sample["target_rgb"].shape == (1, 2, 3, 3)
    assert sample["w2c"].shape == (1, 4, 4)
    assert sample["intrinsic"].shape == (1, 3, 3)
    assert sample["intrinsic"][0, 0, 0] == 2
    assert sample["intrinsic"][0, 2, 2] == 1
    assert sample["height"] == 2
    assert sample["width"] == 3


def test_visual_frame_dataset_reads_ftgspp_memmap_and_rescales_to_requested_scale(tmp_path):
    manifest = _write_manifest(tmp_path)
    visual_root = tmp_path / "visual"
    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3) * 8, np.eye(3) * 10]),
        w2c=np.stack([np.eye(4), np.eye(4) * 2]),
    )
    memmap_root = tmp_path / "memmap"
    memmap_root.mkdir()
    meta = {
        "rgb": {"dtype": "torch.uint8", "shape": [2, 11, 4, 6, 3]},
        "w2c": {"dtype": "torch.float32", "shape": [2, 11, 4, 4]},
        "intrinsic": {"dtype": "torch.float32", "shape": [2, 11, 3, 3]},
        "time": {"dtype": "torch.float32", "shape": [2, 11, 1]},
    }
    (memmap_root / "meta.json").write_text(__import__("json").dumps(meta))
    rgb = np.memmap(memmap_root / "rgb.memmap", mode="w+", dtype=np.uint8, shape=(2, 11, 4, 6, 3))
    rgb[:] = 255
    w2c = np.memmap(memmap_root / "w2c.memmap", mode="w+", dtype=np.float32, shape=(2, 11, 4, 4))
    w2c[:] = 0
    w2c[:, 0] = np.eye(4)
    w2c[:, 10] = np.eye(4) * 3
    intrinsic = np.memmap(
        memmap_root / "intrinsic.memmap",
        mode="w+",
        dtype=np.float32,
        shape=(2, 11, 3, 3),
    )
    intrinsic[:] = 0
    intrinsic[:, 0] = np.eye(3) * 4
    intrinsic[:, 10] = np.eye(3) * 5
    time = np.memmap(memmap_root / "time.memmap", mode="w+", dtype=np.float32, shape=(2, 11, 1))
    time[0, 0, 0] = 0.25
    rgb.flush()
    w2c.flush()
    intrinsic.flush()
    time.flush()

    dataset = VisualFrameDataset(
        manifest,
        split="train",
        scale=0.25,
        memmap_root=memmap_root,
    )

    sample = dataset[0]

    assert sample["target_rgb"].shape == (1, 2, 3, 3)
    assert torch.allclose(sample["target_rgb"], torch.ones(1, 2, 3, 3))
    assert sample["height"] == 2
    assert sample["width"] == 3
    assert sample["time"].item() == pytest.approx(0.25)
    assert sample["intrinsic"][0, 0, 0] == 2
