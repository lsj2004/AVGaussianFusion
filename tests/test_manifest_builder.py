import json
import wave
from pathlib import Path

import pytest

from avfusion.data.build_scene_manifest import build_manifest
from avfusion.data.manifest import SceneManifest


VISUAL_ROOT = Path("/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera")
AUDIO_ROOT = Path("/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera")
REAL_DATA_AVAILABLE = (
    VISUAL_ROOT.exists() and (AUDIO_ROOT / "aligned_16k_stereo" / "near.wav").exists()
)


def _write_wav(path: Path, sample_rate: int = 16000, channels: int = 2) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * (channels * sample_rate // 100))


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="scene1_opera local data is absent")
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


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="scene1_opera local data is absent")
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
        _write_wav(aligned / f"{name}.wav")
    _write_wav(aligned / "near.wav")
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
        _write_wav(aligned / f"{name}.wav")
    _write_wav(aligned / "near.wav")
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


def test_manifest_allows_num_frame_subset_from_visual_manifest(tmp_path):
    visual = tmp_path / "visual"
    audio = tmp_path / "audio"
    aligned = audio / "aligned_16k_stereo"
    visual.mkdir()
    aligned.mkdir(parents=True)
    for name in ("cam00", "cam01"):
        (visual / f"{name}.mp4").write_bytes(b"")
        _write_wav(aligned / f"{name}.wav")
    _write_wav(aligned / "near.wav")
    (visual / "manifest.json").write_text(
        json.dumps({"fps": 24.0, "num_frames": 12, "camera_names": ["cam00", "cam01"]})
    )

    manifest = build_manifest(
        scene_id="toy",
        visual_root=visual,
        audio_root=audio,
        heldout_camera="cam01",
        num_frames=5,
    )

    assert manifest.num_frames == 5
    assert manifest.frame_times == [0.0, 1 / 24.0, 2 / 24.0, 3 / 24.0, 4 / 24.0]


def test_manifest_rejects_num_frame_subset_longer_than_visual_manifest(tmp_path):
    visual = tmp_path / "visual"
    audio = tmp_path / "audio"
    aligned = audio / "aligned_16k_stereo"
    visual.mkdir()
    aligned.mkdir(parents=True)
    for name in ("cam00", "cam01"):
        (visual / f"{name}.mp4").write_bytes(b"")
        _write_wav(aligned / f"{name}.wav")
    _write_wav(aligned / "near.wav")
    (visual / "manifest.json").write_text(
        json.dumps({"fps": 24.0, "num_frames": 12, "camera_names": ["cam00", "cam01"]})
    )

    with pytest.raises(ValueError, match="num_frames exceeds visual manifest"):
        build_manifest(
            scene_id="toy",
            visual_root=visual,
            audio_root=audio,
            heldout_camera="cam01",
            num_frames=13,
        )


def test_manifest_rejects_audio_metadata_mismatch(tmp_path):
    visual = tmp_path / "visual"
    audio = tmp_path / "audio"
    aligned = audio / "aligned_16k_stereo"
    visual.mkdir()
    aligned.mkdir(parents=True)
    for name in ("cam00", "cam01"):
        (visual / f"{name}.mp4").write_bytes(b"")
        _write_wav(aligned / f"{name}.wav")
    _write_wav(aligned / "near.wav")

    with pytest.raises(ValueError, match="sample_rate mismatch"):
        build_manifest(
            scene_id="toy",
            visual_root=visual,
            audio_root=audio,
            heldout_camera="cam01",
            fps=30.0,
            num_frames=10,
            sample_rate=8000,
        )


def test_manifest_requires_visual_timing_without_visual_manifest(tmp_path):
    visual = tmp_path / "visual"
    audio = tmp_path / "audio"
    aligned = audio / "aligned_16k_stereo"
    visual.mkdir()
    aligned.mkdir(parents=True)
    for name in ("cam00", "cam01"):
        (visual / f"{name}.mp4").write_bytes(b"")
        _write_wav(aligned / f"{name}.wav")
    _write_wav(aligned / "near.wav")

    with pytest.raises(ValueError, match="fps must be provided"):
        build_manifest(
            scene_id="toy",
            visual_root=visual,
            audio_root=audio,
            heldout_camera="cam01",
        )
