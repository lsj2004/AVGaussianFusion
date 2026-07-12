import json

import numpy as np
import torch

from avfusion.eval.eval_soft_audio import evaluate_soft_audio_checkpoint
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.soft.acoustic_field import AcousticGaussianField
from avfusion.soft.frequency_transfer_renderer import FrequencyTransferRenderer
from avfusion.train.train_soft_av_gaussians import SoftTrainingConfig, save_soft_checkpoint
from tests.test_audio_video_dataset import _write_manifest
from tests.test_joint_ftgspp_bridge import FakeGaussians


def _write_cameras(tmp_path):
    visual_root = tmp_path / "visual"
    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3), np.eye(3)]),
        w2c=np.stack([np.eye(4), np.eye(4)]),
    )


def _soft_config(manifest, checkpoint):
    return SoftTrainingConfig(
        manifest=str(manifest),
        ftgspp_checkpoint=str(checkpoint),
        output="soft.pt",
        acoustic_steps=1,
        joint_steps=1,
        num_acoustic_points=4,
        top_k=4,
        audio_lr=1e-3,
        acoustic_lr=1e-3,
        shared_lr=0.0,
        audio_window_seconds=0.5,
        audio_crop_mode="center",
        anchored_fraction=0.6,
        dynamic_fraction=0.2,
        anchor_weight=0.1,
        motion_weight=0.1,
        activity_weight=0.1,
        sparse_weight=0.1,
        rgb_weight=0.0,
        audio_weight=1.0,
        visual_guard_psnr_drop_db=0.3,
        audio_guard_relative_drop=0.05,
    )


def test_evaluate_soft_audio_checkpoint_writes_metrics(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
    _write_cameras(tmp_path)
    ftgspp_checkpoint = tmp_path / "gaussians.pt"
    ftgspp_checkpoint.write_bytes(b"placeholder")
    checkpoint = tmp_path / "soft.pt"
    bridge = FTGSRendererBridge(FakeGaussians())
    field = AcousticGaussianField.from_visual_state(bridge.query_state(torch.tensor([[0.0]])), num_points=4)
    renderer = FrequencyTransferRenderer()
    save_soft_checkpoint(
        output_path=checkpoint,
        acoustic_field=field,
        audio_renderer=renderer,
        ftgspp_checkpoint=ftgspp_checkpoint,
        config=_soft_config(manifest, ftgspp_checkpoint),
        loss_history=[{"total": 0.1}],
    )

    metric_lengths = []

    def fake_metrics(pred, target, sample_rate, include_dpam=True):
        metric_lengths.append(pred.shape[-1])
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

    monkeypatch.setattr("avfusion.eval.eval_soft_audio.compute_audiogs_metrics", fake_metrics)

    summary = evaluate_soft_audio_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "eval",
        scale=1.0,
        frame_reader=lambda path, frame_idx: torch.zeros(4, 6, 3),
        max_frames=2,
    )

    assert summary["route"] == "C_soft_av_gaussians"
    assert summary["renderer_type"] == "frequency_transfer"
    assert summary["stage"] == "unknown"
    assert summary["camera"] == "cam10"
    assert summary["num_windows"] == 2
    assert summary["requested_windows"] == 2
    assert summary["skipped_padding_windows"] == 0
    assert summary["dpam_num_windows"] == 0
    assert summary["audio_window_seconds"] == 0.5
    assert summary["audio_eval_protocol"] == "visual_center"
    assert metric_lengths == [8000, 8000]
    assert summary["MAG"] == 1.0
    assert json.loads((tmp_path / "eval" / "audio_summary.json").read_text())["route"] == "C_soft_av_gaussians"


def test_evaluate_soft_audio_checkpoint_rejects_route_b_checkpoint(tmp_path):
    checkpoint = tmp_path / "bad.pt"
    torch.save({"route": "B_joint_av"}, checkpoint)

    try:
        evaluate_soft_audio_checkpoint(
            manifest_path=tmp_path / "manifest.json",
            checkpoint_path=checkpoint,
            output_dir=tmp_path / "eval",
        )
    except ValueError as exc:
        assert "Route C" in str(exc)
    else:
        raise AssertionError("Route B checkpoint should be rejected by Route C evaluator")
