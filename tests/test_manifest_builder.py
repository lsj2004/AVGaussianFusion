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
