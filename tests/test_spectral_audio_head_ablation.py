from pathlib import Path

import pytest
import yaml

from avfusion.train.train_joint_av_gaussians import (
    build_arg_parser,
    build_model,
    resolve_training_config,
)
from tests.test_joint_ftgspp_bridge import FakeGaussians


def test_route_b_config_defaults_to_simple_audio_head(tmp_path):
    config_path = tmp_path / "route_b.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  manifest: /data/scene_manifest.json",
                "  ftgspp_checkpoint: /data/gaussians.pt",
                "  output_checkpoint: /out/joint.pt",
                "train:",
                "  warmup_steps: 0",
                "  joint_steps: 2",
                "  top_k: 5",
                "  audio_lr: 0.003",
                "  shared_lr: 0.0004",
            ]
        )
        + "\n"
    )

    args = build_arg_parser().parse_args(["--config", str(config_path)])
    cfg = resolve_training_config(args)

    assert cfg.audio_head_type == "simple"


def test_route_b_config_can_select_spectral_audio_head(tmp_path):
    config_path = tmp_path / "route_b_spectral.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  manifest: /data/scene_manifest.json",
                "  ftgspp_checkpoint: /data/gaussians.pt",
                "  output_checkpoint: /out/joint.pt",
                "train:",
                "  warmup_steps: 0",
                "  joint_steps: 2",
                "  top_k: 5",
                "  audio_lr: 0.003",
                "  shared_lr: 0.0004",
                "  audio_head_type: spectral",
            ]
        )
        + "\n"
    )

    args = build_arg_parser().parse_args(["--config", str(config_path)])
    cfg = resolve_training_config(args)

    assert cfg.audio_head_type == "spectral"


def test_build_model_can_create_spectral_audio_head(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "ftgs.pt"

    def fake_load_checkpoint(path):
        assert path == checkpoint_path
        from avfusion.joint.ftgspp_bridge import FTGSRendererBridge

        return FTGSRendererBridge(FakeGaussians())

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        fake_load_checkpoint,
    )

    model = build_model(checkpoint_path, top_k=3, audio_head_type="spectral")

    assert type(model.audio_head).__name__ == "SpectralJointAudioHead"
    assert model.audio_head.active_count == 3


def test_build_model_rejects_unknown_audio_head_type(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "ftgs.pt"

    def fake_load_checkpoint(path):
        assert path == checkpoint_path
        from avfusion.joint.ftgspp_bridge import FTGSRendererBridge

        return FTGSRendererBridge(FakeGaussians())

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        fake_load_checkpoint,
    )

    with pytest.raises(ValueError, match="unknown audio_head_type"):
        build_model(checkpoint_path, top_k=3, audio_head_type="bogus")


def test_spectral_head_ablation_configs_and_scripts_are_isolated():
    expected = {
        "scene1_opera_b_joint_av_spectral_no_warmup.yaml": "scene1_opera_b_joint_av_spectral_no_warmup",
        "scene7_playing_300_b_joint_av_spectral_no_warmup.yaml": "scene7_playing_300_b_joint_av_spectral_no_warmup",
    }
    for filename, run_name in expected.items():
        cfg = yaml.safe_load((Path("configs") / filename).read_text())
        assert cfg["route"] == "B_joint_av_spectral_no_warmup"
        assert cfg["paths"]["output_dir"].endswith(f"/runs/{run_name}")
        assert cfg["paths"]["output_checkpoint"].endswith(f"/runs/{run_name}/joint_finetune.pt")
        assert cfg["train"]["warmup_steps"] == 0
        assert cfg["train"]["joint_steps"] == 1000
        assert cfg["train"]["audio_window_seconds"] == 0.5
        assert cfg["train"]["audio_head_type"] == "spectral"

    for script_name in [
        "train_joint_spectral_no_warmup_scene1_opera.sh",
        "eval_joint_spectral_no_warmup_scene1_opera.sh",
        "train_joint_spectral_no_warmup_scene7_playing_300.sh",
        "eval_joint_spectral_no_warmup_scene7_playing_300.sh",
    ]:
        text = (Path("scripts") / script_name).read_text()
        assert "set -euo pipefail" in text
        assert "spectral_no_warmup" in text
        assert "PYTHONPATH=\"${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}\"" in text
