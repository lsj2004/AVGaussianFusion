import torch

from avfusion.soft.acoustic_field import AcousticGaussianField
from avfusion.train.train_soft_av_gaussians import (
    SoftTrainingConfig,
    build_arg_parser,
    compute_total_soft_loss,
    coupling_scale_for_step,
    resolve_soft_training_config,
)


def _state(num_points=5):
    xyz = torch.randn(num_points, 3)
    return {
        "xyz": xyz,
        "opacity": torch.full((num_points, 1), 0.5),
        "velocity": torch.zeros(num_points, 3),
    }


def test_soft_config_defaults_to_short_centered_audio_window(tmp_path):
    config_path = tmp_path / "route_c.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  manifest: /data/scene_manifest.json",
                "  ftgspp_checkpoint: /data/gaussians.pt",
                "  output_checkpoint: /out/soft.pt",
                "train:",
                "  acoustic_steps: 3",
                "  joint_steps: 4",
                "  num_acoustic_points: 9",
                "  audio_lr: 0.002",
            ]
        )
        + "\n"
    )

    cfg = resolve_soft_training_config(build_arg_parser().parse_args(["--config", str(config_path)]))

    assert cfg.manifest == "/data/scene_manifest.json"
    assert cfg.ftgspp_checkpoint == "/data/gaussians.pt"
    assert cfg.output == "/out/soft.pt"
    assert cfg.acoustic_steps == 3
    assert cfg.joint_steps == 4
    assert cfg.num_acoustic_points == 9
    assert cfg.audio_window_seconds == 0.5
    assert cfg.audio_crop_mode == "center"
    assert cfg.anchored_fraction == 0.6
    assert cfg.dynamic_fraction == 0.2
    assert cfg.audio_lr == 0.002
    assert cfg.visual_guard_psnr_drop_db == 0.3
    assert cfg.audio_guard_relative_drop == 0.05


def test_soft_config_reads_tf_diff_ratio_loss_controls(tmp_path):
    config_path = tmp_path / "route_c.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  manifest: /data/scene_manifest.json",
                "  ftgspp_checkpoint: /data/gaussians.pt",
                "  output_checkpoint: /out/soft.pt",
                "losses:",
                "  audio_lre_loss_weight: 0.2",
                "  audio_tf_diff_ratio_loss_weight: 0.05",
                "  audio_tf_diff_ratio_margin_db: 1.0",
            ]
        )
        + "\n"
    )

    cfg = resolve_soft_training_config(build_arg_parser().parse_args(["--config", str(config_path)]))

    assert cfg.audio_lre_loss_weight == 0.2
    assert cfg.audio_tf_diff_ratio_loss_weight == 0.05
    assert cfg.audio_tf_diff_ratio_margin_db == 1.0


def test_soft_config_rejects_long_default_route_c_window(tmp_path):
    config_path = tmp_path / "route_c.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  manifest: /data/scene_manifest.json",
                "  ftgspp_checkpoint: /data/gaussians.pt",
                "  output_checkpoint: /out/soft.pt",
                "train:",
                "  audio_window_seconds: 3.0",
            ]
        )
        + "\n"
    )

    try:
        resolve_soft_training_config(build_arg_parser().parse_args(["--config", str(config_path)]))
    except ValueError as exc:
        assert "audio_window_seconds" in str(exc)
        assert "quasi-static" in str(exc)
    else:
        raise AssertionError("long Route C audio window should be rejected by default")


def test_compute_total_soft_loss_applies_coupling_weights_and_backpropagates():
    visual_state = _state(5)
    field = AcousticGaussianField.from_visual_state(visual_state, num_points=6)
    acoustic_state = field.query(torch.tensor([[0.0]]))
    audio_loss = acoustic_state["xyz"].pow(2).mean()
    rgb_loss = acoustic_state["opacity"].mean() * 0.0
    cfg = SoftTrainingConfig(
        manifest="manifest.json",
        ftgspp_checkpoint="gaussians.pt",
        output="soft.pt",
        acoustic_steps=1,
        joint_steps=1,
        num_acoustic_points=6,
        top_k=4,
        audio_lr=1e-3,
        acoustic_lr=1e-3,
        shared_lr=0.0,
        audio_window_seconds=0.5,
        audio_crop_mode="center",
        anchored_fraction=0.6,
        dynamic_fraction=0.2,
        anchor_weight=0.1,
        motion_weight=0.2,
        activity_weight=0.3,
        sparse_weight=0.4,
        rgb_weight=0.0,
        audio_weight=1.0,
        visual_guard_psnr_drop_db=0.3,
        audio_guard_relative_drop=0.05,
    )

    losses = compute_total_soft_loss(
        cfg,
        field,
        acoustic_state,
        visual_state,
        audio_loss=audio_loss,
        rgb_loss=rgb_loss,
    )
    losses["total"].backward()

    assert losses["total"] > audio_loss.detach()
    assert losses["coupling_scale"] == 1.0
    assert field.means.grad is not None


def test_coupling_schedule_preserves_audio_warmup_before_joint_stage():
    assert coupling_scale_for_step(0, acoustic_steps=3, joint_steps=4) == 0.0
    assert coupling_scale_for_step(2, acoustic_steps=3, joint_steps=4) == 0.0
    assert coupling_scale_for_step(3, acoustic_steps=3, joint_steps=4) == 0.25
    assert coupling_scale_for_step(6, acoustic_steps=3, joint_steps=4) == 1.0
