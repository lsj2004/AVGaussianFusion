import json
from pathlib import Path

import torch

from avfusion.eval.eval_joint_audio import evaluate_joint_audio_checkpoint
from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.model import JointAVGaussianModel
from avfusion.train.train_joint_av_gaussians import save_joint_checkpoint
from tests.test_audio_video_dataset import _write_manifest
from tests.test_joint_ftgspp_bridge import FakeGaussians


def test_evaluate_joint_audio_checkpoint_writes_audiogs_metrics(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
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
        config={"top_k": 4},
        loss_history=[0.1],
    )

    def fake_load_checkpoint(path):
        assert Path(path) == ftgspp_checkpoint
        return FTGSRendererBridge(FakeGaussians())

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        fake_load_checkpoint,
    )

    summary = evaluate_joint_audio_checkpoint(
        manifest_path=manifest,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
    )

    assert summary["camera"] == "cam10"
    assert summary["stage"] == "audio_warmup"
    assert "MAG" in summary
    assert "ENV" in summary
    assert "LRE" in summary
    assert "RTE" in summary
    assert "DPAM" in summary
    assert "debug" in summary
    assert "l1_waveform" in summary["debug"]
    assert json.loads((output_dir / "audio_summary.json").read_text())["camera"] == "cam10"
