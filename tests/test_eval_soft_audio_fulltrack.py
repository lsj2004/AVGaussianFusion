import json

import soundfile as sf
import torch

from avfusion.eval.eval_soft_audio_fulltrack import evaluate_soft_audio_fulltrack_checkpoint
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.soft.acoustic_field import AcousticGaussianField
from avfusion.soft.frequency_transfer_renderer import FrequencyTransferRenderer
from avfusion.train.train_soft_av_gaussians import save_soft_checkpoint
from tests.test_eval_soft_audio import _soft_config, _write_cameras
from tests.test_audio_video_dataset import _write_manifest
from tests.test_joint_ftgspp_bridge import FakeGaussians


def _write_soft_checkpoint(tmp_path):
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
    return manifest, checkpoint


def test_evaluate_soft_audio_fulltrack_writes_global_metrics(monkeypatch, tmp_path):
    manifest, checkpoint = _write_soft_checkpoint(tmp_path)
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

    monkeypatch.setattr("avfusion.eval.eval_soft_audio_fulltrack.compute_audiogs_metrics", fake_metrics)

    summary = evaluate_soft_audio_fulltrack_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "fulltrack",
        protocol="audiogs_3s_nonoverlap_fulltrack",
        audio_window_seconds=0.5,
        max_windows=2,
        scale=1.0,
        frame_reader=lambda path, frame_idx: torch.zeros(4, 6, 3),
    )

    assert summary["route"] == "C_soft_av_gaussians"
    assert summary["eval_type"] == "fulltrack"
    assert summary["protocol"] == "audiogs_3s_nonoverlap_fulltrack"
    assert summary["num_windows"] == 1
    assert summary["audio_window_seconds"] == 0.5
    assert summary["num_samples"] == 8000
    assert summary["target_lr_db"] == summary["target_lr_db"]
    assert summary["pred_lr_db"] == summary["pred_lr_db"]
    assert summary["lr_db_error"] == summary["lr_db_error"]
    assert metric_lengths == [8000]
    assert json.loads((tmp_path / "fulltrack" / "full_audio_summary.json").read_text())["MAG"] == 0.1
    assert (tmp_path / "fulltrack" / "pred_full.wav").exists()
    assert (tmp_path / "fulltrack" / "target_full.wav").exists()
    pred_audio, sample_rate = sf.read(tmp_path / "fulltrack" / "pred_full.wav", always_2d=True)
    assert sample_rate == 16000
    assert pred_audio.shape == (8000, 2)


def test_evaluate_soft_audio_fulltrack_rejects_unknown_protocol(tmp_path):
    manifest, checkpoint = _write_soft_checkpoint(tmp_path)

    try:
        evaluate_soft_audio_fulltrack_checkpoint(
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            output_dir=tmp_path / "fulltrack",
            protocol="bad_protocol",
            frame_reader=lambda path, frame_idx: torch.zeros(4, 6, 3),
        )
    except ValueError as exc:
        assert "unknown fulltrack protocol" in str(exc)
    else:
        raise AssertionError("unknown fulltrack protocol should be rejected")
