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
