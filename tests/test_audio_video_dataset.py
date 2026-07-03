import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from avfusion.data.audio_video_dataset import AudioCropDataset
from avfusion.data.build_scene_manifest import build_manifest


def _write_wav(path: Path, frames: int = 8000) -> None:
    audio = np.zeros((frames, 2), dtype=np.float32)
    sf.write(path, audio, 16000)


def _write_manifest(tmp_path: Path) -> Path:
    visual = tmp_path / "visual"
    audio = tmp_path / "audio"
    aligned = audio / "aligned_16k_stereo"
    visual.mkdir()
    aligned.mkdir(parents=True)

    for name in ("cam00", "cam10"):
        (visual / f"{name}.mp4").write_bytes(b"")
        _write_wav(aligned / f"{name}.wav")
    _write_wav(aligned / "near.wav")
    (visual / "manifest.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "num_frames": 150,
                "camera_names": ["cam00", "cam10"],
            }
        )
    )

    manifest_path = tmp_path / "scene_manifest.json"
    build_manifest(
        scene_id="toy",
        visual_root=visual,
        audio_root=audio,
        heldout_camera="cam10",
        output_path=manifest_path,
    )
    return manifest_path


def test_audio_crop_dataset_returns_source_and_target(tmp_path):
    manifest_path = _write_manifest(tmp_path)

    dataset = AudioCropDataset(manifest_path, split="train")
    sample = dataset[0]

    assert sample["camera"] == "cam00"
    assert sample["source_audio"].shape == (2, 48000)
    assert sample["target_audio"].shape == (2, 48000)
    assert sample["source_audio"].dtype == torch.float32
    assert sample["target_audio"].dtype == torch.float32


def test_audio_crop_dataset_eval_uses_heldout_camera(tmp_path):
    manifest_path = _write_manifest(tmp_path)

    dataset = AudioCropDataset(manifest_path, split="eval")

    assert len(dataset) == 1
    assert dataset[0]["camera"] == "cam10"
