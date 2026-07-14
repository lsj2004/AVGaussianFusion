import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from avfusion.eval.eval_joint_audio import evaluate_joint_audio_checkpoint
from avfusion.eval.eval_joint_audio_fulltrack import evaluate_joint_audio_fulltrack_checkpoint
from avfusion.joint.audio_head import JointAudioHead, SpectralJointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.model import JointAVGaussianModel
from avfusion.train.train_joint_av_gaussians import save_joint_checkpoint
from tests.test_audio_video_dataset import _write_manifest, _write_wav
from tests.test_joint_ftgspp_bridge import FakeGaussians


def test_evaluate_joint_audio_checkpoint_writes_audiogs_metrics(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
    visual_root = tmp_path / "visual"
    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3), np.eye(3)]),
        w2c=np.stack([np.eye(4), np.eye(4)]),
    )
    ftgspp_checkpoint = tmp_path / "gaussians.pt"
    checkpoint = tmp_path / "joint.pt"
    output_dir = tmp_path / "eval"
    ftgspp_checkpoint.write_bytes(b"placeholder")
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4, top_k=4))
    save_joint_checkpoint(
        output_path=checkpoint,
        model=model,
        stage="audio_warmup",
        ftgspp_checkpoint=ftgspp_checkpoint,
        manifest_path=manifest,
        config={"top_k": 4, "visual_scale": 0.5, "audio_window_seconds": 0.5},
        loss_history=[0.1],
    )

    def fake_load_checkpoint(path):
        assert Path(path) == ftgspp_checkpoint
        return FTGSRendererBridge(FakeGaussians())

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        fake_load_checkpoint,
    )
    render_times = []
    render_w2cs = []
    original_render_audio = JointAVGaussianModel.render_audio

    def spy_render_audio(self, t, source_audio, camera_w2c=None):
        render_times.append(float(t.detach().cpu().reshape(-1)[0]))
        render_w2cs.append(camera_w2c.detach().cpu().clone() if camera_w2c is not None else None)
        return original_render_audio(self, t, source_audio, camera_w2c=camera_w2c)

    monkeypatch.setattr(JointAVGaussianModel, "render_audio", spy_render_audio)
    metric_lengths = []
    include_dpam_flags = []

    def fake_metrics(pred, target, sample_rate, include_dpam=True):
        metric_lengths.append(pred.shape[-1])
        include_dpam_flags.append(include_dpam)
        idx = len(metric_lengths)
        return {
            "MAG": float(idx),
            "ENV": float(idx + 2),
            "LRE": float(idx + 4),
            "RTE": None,
            "RTE_available": False,
            "RTE_error": "not configured",
            "DPAM": None,
            "DPAM_available": False,
            "DPAM_error": "not configured",
        }

    monkeypatch.setattr("avfusion.eval.eval_joint_audio.compute_audiogs_metrics", fake_metrics)

    summary = evaluate_joint_audio_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        frame_reader=lambda path, frame_idx: torch.zeros(4, 6, 3),
        max_frames=2,
    )

    assert summary["camera"] == "cam10"
    assert summary["stage"] == "audio_warmup"
    assert summary["num_windows"] == 2
    assert summary["audio_window_seconds"] == 0.5
    assert render_times == pytest.approx([0.0, 1 / 30])
    assert len(render_w2cs) == 2
    assert all(w2c is not None and w2c.shape == (1, 4, 4) for w2c in render_w2cs)
    assert metric_lengths == [8000, 8000]
    assert include_dpam_flags == [False, False]
    assert summary["MAG"] == pytest.approx(1.5)
    assert summary["ENV"] == pytest.approx(3.5)
    assert summary["LRE"] == pytest.approx(5.5)
    assert "RTE" in summary
    assert "DPAM" in summary
    assert "window_metrics" in summary
    assert len(summary["window_metrics"]) == 2
    assert "concatenated_overlapping_debug" in summary
    assert "debug" not in summary
    assert "l1_waveform" in summary["concatenated_overlapping_debug"]
    assert json.loads((output_dir / "audio_summary.json").read_text())["camera"] == "cam10"


def test_evaluate_joint_audio_checkpoint_can_sample_dpam_windows(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
    visual_root = tmp_path / "visual"
    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3), np.eye(3)]),
        w2c=np.stack([np.eye(4), np.eye(4)]),
    )
    ftgspp_checkpoint = tmp_path / "gaussians.pt"
    checkpoint = tmp_path / "joint.pt"
    output_dir = tmp_path / "eval"
    ftgspp_checkpoint.write_bytes(b"placeholder")
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4, top_k=4))
    save_joint_checkpoint(
        output_path=checkpoint,
        model=model,
        stage="joint_finetune",
        ftgspp_checkpoint=ftgspp_checkpoint,
        manifest_path=manifest,
        config={"top_k": 4, "visual_scale": 0.5, "audio_window_seconds": 0.5},
        loss_history=[0.1],
    )

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        lambda path: FTGSRendererBridge(FakeGaussians()),
    )
    include_dpam_flags = []

    def fake_metrics(pred, target, sample_rate, include_dpam=True):
        include_dpam_flags.append(include_dpam)
        idx = len(include_dpam_flags)
        return {
            "MAG": float(idx),
            "ENV": float(idx),
            "LRE": float(idx),
            "RTE": None,
            "RTE_available": False,
            "RTE_error": "not configured",
            "DPAM": float(idx) if include_dpam else None,
            "DPAM_available": bool(include_dpam),
            "DPAM_error": None if include_dpam else "not requested",
        }

    monkeypatch.setattr("avfusion.eval.eval_joint_audio.compute_audiogs_metrics", fake_metrics)

    summary = evaluate_joint_audio_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        frame_reader=lambda path, frame_idx: torch.zeros(4, 6, 3),
        max_frames=4,
        include_dpam=True,
        dpam_max_windows=2,
    )

    assert include_dpam_flags == [False, False, False, False, True, True]
    assert summary["MAG"] == pytest.approx(2.5)
    assert summary["DPAM"] == pytest.approx(5.5)
    assert summary["DPAM_available"] is True
    assert summary["dpam_num_windows"] == 2


def test_evaluate_joint_audio_checkpoint_can_skip_padding_windows(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
    aligned = tmp_path / "audio" / "aligned_16k_stereo"
    _write_wav(aligned / "near.wav", frames=16000, value=0.5)
    _write_wav(aligned / "cam10.wav", frames=16000, value=0.25)
    visual_root = tmp_path / "visual"
    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3), np.eye(3)]),
        w2c=np.stack([np.eye(4), np.eye(4)]),
    )
    ftgspp_checkpoint = tmp_path / "gaussians.pt"
    checkpoint = tmp_path / "joint.pt"
    output_dir = tmp_path / "eval"
    ftgspp_checkpoint.write_bytes(b"placeholder")
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4, top_k=4))
    save_joint_checkpoint(
        output_path=checkpoint,
        model=model,
        stage="joint_finetune",
        ftgspp_checkpoint=ftgspp_checkpoint,
        manifest_path=manifest,
        config={"top_k": 4, "visual_scale": 0.5, "audio_window_seconds": 0.5},
        loss_history=[0.1],
    )
    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        lambda path: FTGSRendererBridge(FakeGaussians()),
    )
    monkeypatch.setattr(
        "avfusion.eval.eval_joint_audio.compute_audiogs_metrics",
        lambda pred, target, sample_rate, include_dpam=True: {
            "MAG": 1.0,
            "ENV": 2.0,
            "LRE": 3.0,
            "RTE": None,
            "RTE_available": False,
            "RTE_error": "not configured",
            "DPAM": None,
            "DPAM_available": False,
            "DPAM_error": "not configured",
        },
    )

    summary = evaluate_joint_audio_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        frame_reader=lambda path, frame_idx: torch.zeros(4, 6, 3),
        max_frames=20,
        skip_padding_windows=True,
    )

    assert summary["audio_eval_protocol"] == "visual_center_skip_padding"
    assert summary["num_windows"] == 12
    assert summary["skipped_padding_windows"] == 8
    assert [row["frame"] for row in summary["window_metrics"]][:3] == [8, 9, 10]
    assert all(row["start_sample"] >= 0 for row in summary["window_metrics"])


def test_evaluate_joint_audio_checkpoint_supports_audiogs_start_nonoverlap(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
    aligned = tmp_path / "audio" / "aligned_16k_stereo"
    _write_wav(aligned / "near.wav", frames=48000, value=0.5)
    _write_wav(aligned / "cam10.wav", frames=48000, value=0.25)
    visual_root = tmp_path / "visual"
    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3), np.eye(3)]),
        w2c=np.stack([np.eye(4), np.eye(4)]),
    )
    ftgspp_checkpoint = tmp_path / "gaussians.pt"
    checkpoint = tmp_path / "joint.pt"
    output_dir = tmp_path / "eval"
    ftgspp_checkpoint.write_bytes(b"placeholder")
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4, top_k=4))
    save_joint_checkpoint(
        output_path=checkpoint,
        model=model,
        stage="joint_finetune",
        ftgspp_checkpoint=ftgspp_checkpoint,
        manifest_path=manifest,
        config={"top_k": 4, "visual_scale": 0.5, "audio_window_seconds": 0.5},
        loss_history=[0.1],
    )
    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        lambda path: FTGSRendererBridge(FakeGaussians()),
    )
    monkeypatch.setattr(
        "avfusion.eval.eval_joint_audio.compute_audiogs_metrics",
        lambda pred, target, sample_rate, include_dpam=True: {
            "MAG": 1.0,
            "ENV": 2.0,
            "LRE": 3.0,
            "RTE": None,
            "RTE_available": False,
            "RTE_error": "not configured",
            "DPAM": None,
            "DPAM_available": False,
            "DPAM_error": "not configured",
        },
    )

    summary = evaluate_joint_audio_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        frame_reader=lambda path, frame_idx: torch.zeros(4, 6, 3),
        audio_eval_protocol="audiogs_start_nonoverlap",
        include_dpam=True,
    )

    assert summary["audio_eval_protocol"] == "audiogs_start_nonoverlap"
    assert summary["num_windows"] == 6
    assert summary["dpam_num_windows"] == 6
    assert [row["frame"] for row in summary["window_metrics"]] == [0, 15, 30, 45, 60, 75]
    assert [row["start_sample"] for row in summary["window_metrics"]] == [
        0,
        8000,
        16000,
        24000,
        32000,
        40000,
    ]


def test_evaluate_joint_audio_checkpoint_restores_spectral_audio_head(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
    visual_root = tmp_path / "visual"
    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3), np.eye(3)]),
        w2c=np.stack([np.eye(4), np.eye(4)]),
    )
    ftgspp_checkpoint = tmp_path / "gaussians.pt"
    checkpoint = tmp_path / "joint_spectral.pt"
    output_dir = tmp_path / "eval"
    ftgspp_checkpoint.write_bytes(b"placeholder")
    model = JointAVGaussianModel(
        FTGSRendererBridge(FakeGaussians()),
        SpectralJointAudioHead(4, top_k=4),
    )
    save_joint_checkpoint(
        output_path=checkpoint,
        model=model,
        stage="joint_finetune",
        ftgspp_checkpoint=ftgspp_checkpoint,
        manifest_path=manifest,
        config={
            "top_k": 4,
            "visual_scale": 0.5,
            "audio_window_seconds": 0.5,
            "audio_head_type": "spectral",
        },
        loss_history=[0.1],
    )

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        lambda path: FTGSRendererBridge(FakeGaussians()),
    )

    def fake_metrics(pred, target, sample_rate, include_dpam=True):
        return {
            "MAG": 1.0,
            "ENV": 2.0,
            "LRE": 3.0,
            "RTE": None,
            "RTE_available": False,
            "RTE_error": "not configured",
            "DPAM": None,
            "DPAM_available": False,
            "DPAM_error": "not configured",
        }

    monkeypatch.setattr("avfusion.eval.eval_joint_audio.compute_audiogs_metrics", fake_metrics)

    summary = evaluate_joint_audio_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        frame_reader=lambda path, frame_idx: torch.zeros(4, 6, 3),
        max_frames=1,
    )

    assert summary["MAG"] == pytest.approx(1.0)
    assert json.loads((output_dir / "audio_summary.json").read_text())["checkpoint"] == str(checkpoint)


def test_evaluate_joint_audio_fulltrack_writes_global_metrics(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
    aligned = tmp_path / "audio" / "aligned_16k_stereo"
    _write_wav(aligned / "near.wav", frames=48000, value=0.5)
    _write_wav(aligned / "cam10.wav", frames=48000, value=0.25)
    visual_root = tmp_path / "visual"
    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3), np.eye(3)]),
        w2c=np.stack([np.eye(4), np.eye(4)]),
    )
    ftgspp_checkpoint = tmp_path / "gaussians.pt"
    checkpoint = tmp_path / "joint.pt"
    output_dir = tmp_path / "fulltrack"
    ftgspp_checkpoint.write_bytes(b"placeholder")
    model = JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4, top_k=4))
    save_joint_checkpoint(
        output_path=checkpoint,
        model=model,
        stage="joint_finetune",
        ftgspp_checkpoint=ftgspp_checkpoint,
        manifest_path=manifest,
        config={"top_k": 4, "visual_scale": 0.5, "audio_window_seconds": 0.5},
        loss_history=[0.1],
    )
    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        lambda path: FTGSRendererBridge(FakeGaussians()),
    )
    metric_lengths = []

    def fake_metrics(pred, target, sample_rate, include_dpam=True):
        metric_lengths.append(pred.shape[-1])
        return {
            "MAG": 0.1,
            "ENV": 0.2,
            "LRE": 0.3,
            "RTE": None,
            "RTE_available": False,
            "RTE_error": "not configured",
            "DPAM": None,
            "DPAM_available": False,
            "DPAM_error": "not configured",
        }

    monkeypatch.setattr("avfusion.eval.eval_joint_audio_fulltrack.compute_audiogs_metrics", fake_metrics)

    summary = evaluate_joint_audio_fulltrack_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        protocol="audiogs_3s_nonoverlap_fulltrack",
        audio_window_seconds=0.5,
        max_windows=2,
        frame_reader=lambda path, frame_idx: torch.zeros(4, 6, 3),
    )

    assert summary["route"] == "B_joint_av"
    assert summary["eval_type"] == "fulltrack"
    assert summary["protocol"] == "audiogs_3s_nonoverlap_fulltrack"
    assert summary["num_windows"] == 2
    assert summary["audio_window_seconds"] == 0.5
    assert summary["num_samples"] == 16000
    assert metric_lengths == [16000]
    assert json.loads((output_dir / "full_audio_summary.json").read_text())["MAG"] == 0.1
    pred_audio, sample_rate = sf.read(output_dir / "pred_full.wav", always_2d=True)
    assert sample_rate == 16000
    assert pred_audio.shape == (16000, 2)
