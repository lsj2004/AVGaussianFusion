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
    assert cfg["train"]["audio_window_seconds"] == 0.5
    assert cfg["losses"]["visual_weight"] == 1.0
    assert cfg["losses"]["audio_weight"] == 1.0
    assert cfg["losses"]["geometry_reg_weight"] == 0.001


def test_scene1_route_b_no_warmup_config_targets_separate_outputs():
    cfg_path = Path("configs/scene1_opera_b_joint_av_no_warmup.yaml")
    assert cfg_path.exists()
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["route"] == "B_joint_av_no_warmup"
    assert cfg["scene"] == "scene1_opera"
    assert cfg["paths"]["manifest"] == "/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/scene_manifest.json"
    assert cfg["paths"]["ftgspp_checkpoint"] == "/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/ftgspp/scene1_opera/00/gaussians.pt"
    assert cfg["paths"]["ftgspp_memmap"] == "/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus/_memmap/avgaussianfusion_scene1_opera_a/scene1_opera"
    assert cfg["paths"]["output_dir"] == "/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_b_joint_av_no_warmup"
    assert cfg["paths"]["output_checkpoint"] == "/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_b_joint_av_no_warmup/joint_finetune.pt"
    assert cfg["train"]["warmup_steps"] == 0
    assert cfg["train"]["joint_steps"] == 1000
    assert cfg["losses"]["visual_weight"] == 1.0
    assert cfg["losses"]["audio_weight"] == 1.0


def test_scene1_route_b_scripts_document_train_and_eval_entrypoints():
    train_script = Path("scripts/train_joint_scene1_opera.sh").read_text()
    eval_script = Path("scripts/eval_joint_scene1_opera.sh").read_text()
    train_no_warmup_script = Path("scripts/train_joint_no_warmup_scene1_opera.sh").read_text()
    eval_no_warmup_script = Path("scripts/eval_joint_no_warmup_scene1_opera.sh").read_text()

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

    assert "scene1_opera_b_joint_av_no_warmup.yaml" in train_no_warmup_script
    assert "runs/scene1_opera_b_joint_av_no_warmup" in train_no_warmup_script
    assert "python -m avfusion.train.train_joint_av_gaussians" in train_no_warmup_script
    assert "runs/scene1_opera_b_joint_av_no_warmup" in eval_no_warmup_script
    assert "python -m avfusion.eval.eval_joint_audio" in eval_no_warmup_script
    assert "python -m avfusion.eval.eval_joint_visual" in eval_no_warmup_script


def test_readme_labels_route_b_eval_as_metric_entrypoint():
    readme = Path("README.md").read_text()

    assert "scripts/eval_joint_scene1_opera.sh" in readme
    assert "scripts/train_joint_no_warmup_scene1_opera.sh" in readme
    assert "AudioGS-style heldout audio metrics" in readme


def test_fair_comparison_configs_target_separate_outputs():
    expected = {
        "scene1_opera_b_joint_av_fair.yaml": (
            "scene1_opera_b_joint_av_fair",
            1000,
        ),
        "scene1_opera_b_joint_av_no_warmup_fair.yaml": (
            "scene1_opera_b_joint_av_no_warmup_fair",
            0,
        ),
        "scene7_playing_300_b_joint_av_fair.yaml": (
            "scene7_playing_300_b_joint_av_fair",
            1000,
        ),
        "scene7_playing_300_b_joint_av_no_warmup_fair.yaml": (
            "scene7_playing_300_b_joint_av_no_warmup_fair",
            0,
        ),
    }
    for filename, (run_name, warmup_steps) in expected.items():
        cfg = yaml.safe_load((Path("configs") / filename).read_text())
        assert cfg["paths"]["output_dir"].endswith(f"/runs/{run_name}")
        assert cfg["paths"]["output_checkpoint"].endswith(f"/runs/{run_name}/joint_finetune.pt")
        assert cfg["train"]["warmup_steps"] == warmup_steps
        assert cfg["train"]["joint_steps"] == 1000
        assert cfg["train"]["audio_window_seconds"] == 0.5


def test_fair_comparison_runner_documents_protocol_and_outputs():
    script = Path("scripts/run_fair_baseline_comparison.sh").read_text()

    assert "fair_baseline_comparison" in script
    assert "--print-valid-joint-steps" in script
    assert "--joint-steps" in script
    assert "scene1_opera_b_joint_av_fair.yaml" in script
    assert "scene7_playing_300_b_joint_av_no_warmup_fair.yaml" in script
    assert "runs/fair_baseline_comparison" in script
    assert "/metrics/comparison.md" in script
    assert "strict_audiogs_cam10_allcams_summary.json" in script



def test_route_c_stereo_regularization_configs_use_separate_outputs_and_stronger_stereo_losses():
    expected = {
        "scene1_opera_c_soft_av_gaussians_stereo_reg.yaml": "scene1_opera_c_soft_av_gaussians_stereo_reg",
        "scene7_playing_300_c_soft_av_gaussians_stereo_reg.yaml": "scene7_playing_300_c_soft_av_gaussians_stereo_reg",
    }
    for filename, run_name in expected.items():
        cfg = yaml.safe_load((Path("configs") / filename).read_text())
        assert cfg["route"] == "C_soft_av_gaussians"
        assert cfg["paths"]["output_dir"].endswith(f"/runs/{run_name}")
        assert cfg["paths"]["output_checkpoint"].endswith(f"/runs/{run_name}/soft_av.pt")
        assert cfg["losses"]["audio_lre_loss_weight"] == 0.3
        assert cfg["train"]["audio_renderer_use_phase_delay"] is False
        assert cfg["train"]["audio_long_window_seconds"] == 3.0
        assert cfg["train"]["audio_long_window_fraction"] == 0.7
        assert cfg["train"]["audio_long_crop_mode"] == "start"
        assert cfg["losses"]["audio_tf_diff_ratio_loss_weight"] == 0.1
        assert cfg["train"]["audio_renderer_use_geometry_diff_head"] is True
        assert cfg["train"]["audio_renderer_geometry_diff_scale"] == 0.25
        assert cfg["losses"]["audio_band_lre_loss_weight"] == 0.2
        assert cfg["losses"]["audio_tf_ild_loss_weight"] == 0.2
        assert cfg["losses"]["audio_coherence_loss_weight"] == 0.05
        assert cfg["losses"]["audio_phase_diff_loss_weight"] == 0.02
        assert cfg["losses"]["audio_mono_diff_phase_loss_weight"] == 0.02
        assert cfg["losses"]["audio_energy_balance_loss_weight"] == 0.01
        assert cfg["losses"]["diff_directional_response_l2_weight"] == 0.0001
        assert cfg["losses"]["diff_directional_response_smooth_weight"] == 0.001
        assert cfg["losses"]["diff_response_l2_weight"] == 0.0001
        assert cfg["losses"]["diff_response_smooth_weight"] == 0.001
        assert cfg["losses"]["side_response_l2_weight"] == 0.0001
        assert cfg["losses"]["side_response_smooth_weight"] == 0.001


def test_route_c_stereo_regularization_scripts_point_to_ablation_configs():
    scene1_script = Path("scripts/train_soft_av_scene1_opera_stereo_reg.sh").read_text()
    scene7_script = Path("scripts/train_soft_av_scene7_playing_300_stereo_reg.sh").read_text()

    assert "scene1_opera_c_soft_av_gaussians_stereo_reg.yaml" in scene1_script
    assert "scene7_playing_300_c_soft_av_gaussians_stereo_reg.yaml" in scene7_script
    assert "python -m avfusion.train.train_soft_av_gaussians" in scene1_script
    assert "python -m avfusion.train.train_soft_av_gaussians" in scene7_script


def test_route_c_mid_side_phase_configs_enable_phase_delay():
    expected = {
        "scene1_opera_c_soft_av_gaussians_mid_side_phase.yaml": "scene1_opera_c_soft_av_gaussians_mid_side_phase",
        "scene7_playing_300_c_soft_av_gaussians_mid_side_phase.yaml": "scene7_playing_300_c_soft_av_gaussians_mid_side_phase",
    }
    for filename, run_name in expected.items():
        cfg = yaml.safe_load((Path("configs") / filename).read_text())
        assert cfg["route"] == "C_soft_av_gaussians"
        assert cfg["paths"]["output_dir"].endswith(f"/runs/{run_name}")
        assert cfg["paths"]["output_checkpoint"].endswith(f"/runs/{run_name}/soft_av.pt")
        assert cfg["train"]["audio_renderer_use_phase_delay"] is True
        assert cfg["train"]["audio_long_window_seconds"] == 3.0
        assert cfg["train"]["audio_long_window_fraction"] == 0.7
        assert cfg["train"]["audio_long_crop_mode"] == "start"
        assert cfg["train"]["audio_renderer_use_geometry_diff_head"] is True
        assert cfg["train"]["audio_renderer_geometry_diff_scale"] == 0.25
        assert cfg["losses"]["audio_phase_diff_loss_weight"] == 0.05
        assert cfg["losses"]["audio_mono_diff_phase_loss_weight"] == 0.02
        assert cfg["losses"]["audio_band_lre_loss_weight"] == 0.2
        assert cfg["losses"]["audio_tf_ild_loss_weight"] == 0.2
        assert cfg["losses"]["diff_directional_response_l2_weight"] == 0.0001
        assert cfg["losses"]["diff_directional_response_smooth_weight"] == 0.001


def test_route_c_mid_side_phase_scripts_point_to_phase_configs():
    scene1_script = Path("scripts/train_soft_av_scene1_opera_mid_side_phase.sh").read_text()
    scene7_script = Path("scripts/train_soft_av_scene7_playing_300_mid_side_phase.sh").read_text()

    assert "scene1_opera_c_soft_av_gaussians_mid_side_phase.yaml" in scene1_script
    assert "scene7_playing_300_c_soft_av_gaussians_mid_side_phase.yaml" in scene7_script
    assert "python -m avfusion.train.train_soft_av_gaussians" in scene1_script
    assert "python -m avfusion.train.train_soft_av_gaussians" in scene7_script


def test_route_c_mid_side_phase_fulltrack_eval_scripts_write_global_audio_outputs():
    expected = {
        "scripts/eval_soft_av_fulltrack_scene1_opera_mid_side_phase.sh": "scene1_opera_c_soft_av_gaussians_mid_side_phase",
        "scripts/eval_soft_av_fulltrack_scene7_playing_300_mid_side_phase.sh": "scene7_playing_300_c_soft_av_gaussians_mid_side_phase",
    }
    for script_path, run_name in expected.items():
        script = Path(script_path).read_text()
        assert "python -m avfusion.eval.eval_soft_audio_fulltrack" in script
        assert f"runs/{run_name}" in script
        assert "eval_fulltrack_audiogs_3s_nonoverlap" in script
        assert "eval_fulltrack_visual_center_0p5s_ola" in script
        assert "--protocol audiogs_3s_nonoverlap_fulltrack" in script
        assert "--protocol visual_center_overlap_add_fulltrack" in script


def test_route_c_fulltrack_eval_scripts_write_global_audio_outputs():
    scripts = [
        "scripts/eval_soft_av_fulltrack_scene1_opera.sh",
        "scripts/eval_soft_av_fulltrack_scene7_playing_300.sh",
        "scripts/eval_soft_av_fulltrack_scene1_opera_stereo_reg.sh",
        "scripts/eval_soft_av_fulltrack_scene7_playing_300_stereo_reg.sh",
    ]
    for script_path in scripts:
        script = Path(script_path).read_text()
        assert "python -m avfusion.eval.eval_soft_audio_fulltrack" in script
        assert "eval_fulltrack_audiogs_3s_nonoverlap" in script
        assert "eval_fulltrack_visual_center_0p5s_ola" in script
        assert "--protocol audiogs_3s_nonoverlap_fulltrack" in script
        assert "--protocol visual_center_overlap_add_fulltrack" in script
