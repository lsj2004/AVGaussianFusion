import json
from pathlib import Path

import pytest

from avfusion.data.build_scene_manifest import build_manifest
from avfusion.data.manifest import SceneManifest


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


def test_manifest_loads_round_trip_from_temp_scene(tmp_path):
    visual = tmp_path / "visual"
    audio = tmp_path / "audio"
    aligned = audio / "aligned_16k_stereo"
    visual.mkdir()
    aligned.mkdir(parents=True)
    for name in ("cam00", "cam01"):
        (visual / f"{name}.mp4").write_bytes(b"")
        (aligned / f"{name}.wav").write_bytes(b"")
    (aligned / "near.wav").write_bytes(b"")
    (visual / "manifest.json").write_text(
        json.dumps(
            {
                "fps": 24.0,
                "num_frames": 12,
                "camera_names": ["cam00", "cam01"],
            }
        )
    )
    out = tmp_path / "manifest.json"

    manifest = build_manifest(
        scene_id="toy",
        visual_root=visual,
        audio_root=audio,
        heldout_camera="cam01",
        output_path=out,
    )
    loaded = SceneManifest.load(out)

    assert manifest.fps == 24.0
    assert manifest.num_frames == 12
    assert loaded == manifest
    assert loaded.train_cameras == ["cam00"]
    assert loaded.eval_cameras == ["cam01"]


def test_manifest_rejects_visual_metadata_mismatch(tmp_path):
    visual = tmp_path / "visual"
    audio = tmp_path / "audio"
    aligned = audio / "aligned_16k_stereo"
    visual.mkdir()
    aligned.mkdir(parents=True)
    for name in ("cam00", "cam01"):
        (visual / f"{name}.mp4").write_bytes(b"")
        (aligned / f"{name}.wav").write_bytes(b"")
    (aligned / "near.wav").write_bytes(b"")
    (visual / "manifest.json").write_text(
        json.dumps({"fps": 24.0, "num_frames": 12, "camera_names": ["cam00", "cam01"]})
    )

    with pytest.raises(ValueError, match="fps mismatch"):
        build_manifest(
            scene_id="toy",
            visual_root=visual,
            audio_root=audio,
            heldout_camera="cam01",
            fps=30.0,
        )
