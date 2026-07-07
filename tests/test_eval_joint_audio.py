import json
from pathlib import Path

import numpy as np
import pytest
import torch

from avfusion.eval.eval_joint_audio import evaluate_joint_audio_checkpoint
from avfusion.joint.audio_head import JointAudioHead, SpectralJointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.model import JointAVGaussianModel
from avfusion.train.train_joint_av_gaussians import save_joint_checkpoint
from tests.test_audio_video_dataset import _write_manifest
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
    original_render_audio = JointAVGaussianModel.render_audio

    def spy_render_audio(self, t, source_audio):
        render_times.append(float(t.detach().cpu().reshape(-1)[0]))
        return original_render_audio(self, t, source_audio)

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
