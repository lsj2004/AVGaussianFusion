from pathlib import Path

import yaml


def test_scene1_config_has_route_a_defaults():
    cfg_path = Path("configs/scene1_opera_a_frozen_carrier.yaml")
    assert cfg_path.exists()
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["scene"]["id"] == "scene1_opera"
    assert cfg["split"]["heldout_camera"] == "cam10"
    assert cfg["audio"]["source"] == "near.wav"
    assert cfg["audio"]["crop_seconds"] == 3.0
    assert cfg["audio"]["enable_phase_modeling"] is False
    assert cfg["paths"]["visual_root"] == "/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera"
    assert cfg["paths"]["audio_root"] == "/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera"


def test_ftgspp_scene1_config_matches_route_a_data_and_split():
    cfg_path = Path("configs/ftgspp_scene1_opera/scene1_opera.toml")
    text = cfg_path.read_text()

    assert "Sampled_data/v5_0630_dynerf/scene1_opera" in text
    assert "eval_cameras = [10]" in text
    assert "frames = { \"start\" = 0, \"stop\" = 150 }" in text


def test_scene1_route_b_config_targets_joint_av_outputs():
    cfg_path = Path("configs/scene1_opera_b_joint_av.yaml")
    assert cfg_path.exists()
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["route"] == "B_joint_av"
    assert cfg["scene"] == "scene1_opera"
    assert cfg["paths"]["visual_root"] == "/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera"
    assert cfg["paths"]["audio_root"] == "/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera"
    assert cfg["paths"]["manifest"] == "/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/scene_manifest.json"
    assert cfg["paths"]["ftgspp_checkpoint"] == "/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/ftgspp/scene1_opera/00/gaussians.pt"
    assert cfg["paths"]["ftgspp_memmap"] == "/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/_memmap/avgaussianfusion_scene1_opera_a/scene1_opera"
    assert cfg["paths"]["output_checkpoint"] == "/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_b_joint_av/joint_finetune.pt"
    assert cfg["train"]["warmup_steps"] == 1000
    assert cfg["train"]["joint_steps"] == 1000
    assert cfg["train"]["top_k"] == 8192
    assert cfg["train"]["audio_lr"] == 0.0005
    assert cfg["train"]["shared_lr"] == 0.00001
    assert cfg["train"]["visual_scale"] == 0.125
    assert cfg["losses"]["visual_weight"] == 1.0
    assert cfg["losses"]["audio_weight"] == 1.0
    assert cfg["losses"]["geometry_reg_weight"] == 0.001


def test_scene1_route_b_scripts_document_train_and_eval_entrypoints():
    train_script = Path("scripts/train_joint_scene1_opera.sh").read_text()
    eval_script = Path("scripts/eval_joint_scene1_opera.sh").read_text()

    assert "set -euo pipefail" in train_script
    assert "python -m avfusion.train.train_joint_av_gaussians" in train_script
    assert "--config \"${CONFIG}\"" in train_script
    assert "FTGSPP_ROOT=\"/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus\"" in train_script
    assert "PYTHONPATH=\"${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}\"" in train_script
    assert "uv run --no-sync --with soundfile --with pyyaml python -m avfusion.train.train_joint_av_gaussians" in train_script
    assert "runs/scene1_opera_b_joint_av" in train_script
    assert "runs/scene1_opera_a/stage2_audio.pt" not in train_script

    assert "set -euo pipefail" in eval_script
    assert "runs/scene1_opera_b_joint_av/eval" in eval_script
    assert "joint_finetune.pt" in eval_script
    assert "FTGSPP_ROOT=\"/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus\"" in eval_script
    assert "PYTHONPATH=\"${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}\"" in eval_script
    assert "uv run --no-sync" in eval_script
    assert "--with torch" not in eval_script
    assert "python -m avfusion.eval.eval_joint_audio" in eval_script
    assert "python -m avfusion.eval.eval_joint_visual" in eval_script


def test_readme_labels_route_b_eval_as_metric_entrypoint():
    readme = Path("README.md").read_text()

    assert "scripts/eval_joint_scene1_opera.sh" in readme
    assert "AudioGS-style heldout audio metrics" in readme
