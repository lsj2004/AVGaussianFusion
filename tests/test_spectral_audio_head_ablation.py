from pathlib import Path

import pytest
import torch
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


def test_route_b_config_can_select_audiogs_audio_head(tmp_path):
    config_path = tmp_path / "route_b_audiogs.yaml"
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
                "  audio_head_type: audiogs",
            ]
        )
        + "\n"
    )

    args = build_arg_parser().parse_args(["--config", str(config_path)])
    cfg = resolve_training_config(args)

    assert cfg.audio_head_type == "audiogs"


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


def test_build_model_can_create_audiogs_audio_head(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "ftgs.pt"

    def fake_load_checkpoint(path):
        assert path == checkpoint_path
        from avfusion.joint.ftgspp_bridge import FTGSRendererBridge

        return FTGSRendererBridge(FakeGaussians())

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        fake_load_checkpoint,
    )

    model = build_model(checkpoint_path, top_k=3, audio_head_type="audiogs")

    assert type(model.audio_head).__name__ == "AudioGSMaskedSpectralHead"
    assert model.audio_head.active_count == 3


def test_build_model_binds_audiogs_head_to_initial_topk_visual_carrier(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "ftgs.pt"
    fake_gaussians = FakeGaussians()
    fake_gaussians.opacities.data = torch.tensor([[-3.0], [4.0], [1.0], [2.0]])

    def fake_load_checkpoint(path):
        assert path == checkpoint_path
        from avfusion.joint.ftgspp_bridge import FTGSRendererBridge

        return FTGSRendererBridge(fake_gaussians)

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        fake_load_checkpoint,
    )

    model = build_model(checkpoint_path, top_k=2, audio_head_type="audiogs")

    assert model.audio_head.carrier_indices.tolist() == [1, 3]


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
        "scene1_opera_b_joint_av_spectral_no_warmup.yaml": (
            "scene1_opera_b_joint_av_spectral_no_warmup",
            4940,
        ),
        "scene7_playing_300_b_joint_av_spectral_no_warmup.yaml": (
            "scene7_playing_300_b_joint_av_spectral_no_warmup",
            11134,
        ),
    }
    for filename, (run_name, joint_steps) in expected.items():
        cfg = yaml.safe_load((Path("configs") / filename).read_text())
        assert cfg["route"] == "B_joint_av_spectral_no_warmup"
        assert cfg["paths"]["output_dir"].endswith(f"/runs/{run_name}")
        assert cfg["paths"]["output_checkpoint"].endswith(f"/runs/{run_name}/joint_finetune.pt")
        assert cfg["train"]["warmup_steps"] == 0
        assert cfg["train"]["joint_steps"] == joint_steps
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


def test_audiogs_strict_head_configs_use_audio_gs_protocol():
    expected = {
        "scene1_opera_b_joint_av_audiogs_head_strict.yaml": (
            "scene1_opera_b_joint_av_audiogs_head_strict",
            "train_joint_audiogs_head_strict_scene1_opera.sh",
        ),
        "scene7_playing_300_b_joint_av_audiogs_head_strict.yaml": (
            "scene7_playing_300_b_joint_av_audiogs_head_strict",
            "train_joint_audiogs_head_strict_scene7_playing_300.sh",
        ),
    }
    for filename, (run_name, script_name) in expected.items():
        cfg = yaml.safe_load((Path("configs") / filename).read_text())
        losses = cfg["losses"]
        assert cfg["route"] == "B_joint_av_audiogs_head_strict"
        assert cfg["paths"]["output_dir"].endswith(f"/runs/{run_name}")
        assert cfg["train"]["warmup_steps"] == 0
        assert cfg["train"]["audio_window_seconds"] == 3.0
        assert cfg["train"]["audio_head_type"] == "audiogs"
        assert cfg["train"]["audio_loss_type"] == "audiogs_mono_diff"
        assert losses["audio_diff_weight"] == 0.5
        assert losses["audio_lre_loss_weight"] == 0.075
        assert losses["audio_bandpass"] == {
            "enable": True,
            "low_hz": 150.0,
            "high_hz": -1.0,
            "order": 5,
        }

        script = (Path("scripts") / script_name).read_text()
        assert "--audio-window-seconds 3.0" in script
        assert "audiogs_head_strict" in script
        assert "fair_baseline_comparison" not in script


def test_audiogs_strict_all_scene_scripts_dispatch_both_datasets():
    train_all = (Path("scripts") / "train_joint_audiogs_head_strict_all.sh").read_text()
    eval_all = (Path("scripts") / "eval_joint_audiogs_head_strict_all.sh").read_text()
    run_all = (Path("scripts") / "run_joint_audiogs_head_strict_all.sh").read_text()

    for text in [train_all, eval_all]:
        assert "scene1_opera" in text
        assert "scene7_playing_300" in text
        assert "SCENES=" in text

    assert "MODE=" in run_all
    assert "train_eval" in run_all
