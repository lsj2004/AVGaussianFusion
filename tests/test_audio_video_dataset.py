import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import pytest

from avfusion.data.audio_video_dataset import AudioCropDataset, _apply_audiogs_bandpass
from avfusion.data.build_scene_manifest import build_manifest


def _write_wav(
    path: Path,
    frames: int = 8000,
    sample_rate: int = 16000,
    channels: int = 2,
    value: float = 0.0,
) -> None:
    audio = np.full((frames, channels), value, dtype=np.float32)
    sf.write(path, audio, sample_rate)


def _write_manifest(tmp_path: Path) -> Path:
    visual = tmp_path / "visual"
    audio = tmp_path / "audio"
    aligned = audio / "aligned_16k_stereo"
    visual.mkdir()
    aligned.mkdir(parents=True)

    for name in ("cam00", "cam10"):
        (visual / f"{name}.mp4").write_bytes(b"")
        _write_wav(aligned / f"{name}.wav", value=0.25)
    _write_wav(aligned / "near.wav", value=0.5)
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
    assert torch.all(sample["source_audio"][:, :8000] > 0)
    assert torch.all(sample["source_audio"][:, 8000:] == 0)


def test_audio_crop_dataset_eval_uses_heldout_camera(tmp_path):
    manifest_path = _write_manifest(tmp_path)

    dataset = AudioCropDataset(manifest_path, split="eval")

    assert len(dataset) == 1
    assert dataset[0]["camera"] == "cam10"


def test_audio_crop_dataset_rejects_invalid_split(tmp_path):
    manifest_path = _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="split must be train or eval"):
        AudioCropDataset(manifest_path, split="test")


def test_audio_crop_dataset_rejects_sample_rate_drift(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    aligned = tmp_path / "audio" / "aligned_16k_stereo"
    _write_wav(aligned / "cam00.wav", sample_rate=8000)

    dataset = AudioCropDataset(manifest_path, split="train")
    with pytest.raises(ValueError, match="sample rate mismatch"):
        dataset[0]


def test_audio_crop_dataset_rejects_channel_drift(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    aligned = tmp_path / "audio" / "aligned_16k_stereo"
    _write_wav(aligned / "cam00.wav", channels=1)

    dataset = AudioCropDataset(manifest_path, split="train")
    with pytest.raises(ValueError, match="channel mismatch"):
        dataset[0]


def test_timed_audio_cropper_returns_centered_window_with_padding(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    aligned = tmp_path / "audio" / "aligned_16k_stereo"
    sample_rate = 16000
    samples = np.arange(sample_rate, dtype=np.float32) / sample_rate * 0.25
    stereo = np.stack([samples, samples + 1.0], axis=1)
    sf.write(aligned / "near.wav", stereo, sample_rate)
    sf.write(aligned / "cam00.wav", stereo + 0.25, sample_rate)

    from avfusion.data.audio_video_dataset import TimedAudioCropper

    cropper = TimedAudioCropper(manifest_path, crop_seconds=0.5, mode="center")
    crop = cropper.get_crop("cam00", time_seconds=0.25)

    assert crop["camera"] == "cam00"
    assert crop["time"].item() == pytest.approx(0.25)
    assert crop["start_sample"] == 0
    assert crop["source_audio"].shape == (2, 8000)
    assert crop["target_audio"].shape == (2, 8000)
    assert crop["source_audio"][0, 0].item() == pytest.approx(0.0, abs=1e-4)
    assert crop["target_audio"][0, 0].item() == pytest.approx(0.25, abs=1e-4)

    edge_crop = cropper.get_crop("cam00", time_seconds=0.0)

    assert edge_crop["start_sample"] == -4000
    assert torch.all(edge_crop["source_audio"][:, :4000] == 0)
    assert edge_crop["source_audio"][0, 4000].item() == pytest.approx(0.0, abs=1e-4)


def test_timed_audio_cropper_rejects_too_short_stft_window(tmp_path):
    manifest_path = _write_manifest(tmp_path)

    from avfusion.data.audio_video_dataset import TimedAudioCropper

    with pytest.raises(ValueError, match="n_fft|512"):
        TimedAudioCropper(manifest_path, crop_seconds=0.001)


def test_timed_audio_cropper_can_reject_padding_windows(tmp_path):
    manifest_path = _write_manifest(tmp_path)

    from avfusion.data.audio_video_dataset import TimedAudioCropper

    cropper = TimedAudioCropper(manifest_path, crop_seconds=0.5, allow_padding=False)

    with pytest.raises(ValueError, match="padding"):
        cropper.get_crop("cam00", time_seconds=0.0)


def test_audiogs_bandpass_suppresses_low_frequency_component():
    sample_rate = 16000
    t = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    low = torch.sin(2 * torch.pi * 50.0 * t)
    high = torch.sin(2 * torch.pi * 1000.0 * t)
    audio = torch.stack([low + high, low + high])

    filtered = _apply_audiogs_bandpass(
        audio,
        sample_rate,
        low_hz=150.0,
        high_hz=-1.0,
        order=5,
    )
    low_basis = low / low.norm()
    high_basis = high / high.norm()

    low_energy = torch.matmul(filtered[0], low_basis).abs()
    high_energy = torch.matmul(filtered[0], high_basis).abs()
    assert high_energy > 100 * low_energy
