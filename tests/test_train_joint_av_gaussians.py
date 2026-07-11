import json

import pytest
import torch

from avfusion.train.train_joint_av_gaussians import (
    build_arg_parser,
    build_model,
    resolve_training_config,
    save_joint_checkpoint,
    train_audio_warmup,
    train_and_save,
    train_joint_finetune,
)
from tests.test_audio_video_dataset import _write_manifest, _write_wav
from tests.test_joint_ftgspp_bridge import FakeGaussians


def test_parser_accepts_required_paths_and_numeric_values():
    args = build_arg_parser().parse_args(
        [
            "--manifest",
            "manifest.json",
            "--ftgspp-checkpoint",
            "stage1.pt",
            "--output",
            "joint.pt",
            "--warmup-steps",
            "12",
            "--joint-steps",
            "34",
            "--top-k",
            "3",
            "--audio-lr",
            "0.001",
            "--shared-lr",
            "0.0002",
        ]
    )

    assert args.manifest == "manifest.json"
    assert args.ftgspp_checkpoint == "stage1.pt"
    assert args.output == "joint.pt"
    assert args.warmup_steps == 12
    assert args.joint_steps == 34
    assert args.top_k == 3
    assert args.audio_lr == pytest.approx(0.001)
    assert args.shared_lr == pytest.approx(0.0002)


def test_config_file_populates_route_b_training_args(tmp_path):
    config_path = tmp_path / "route_b.yaml"
    config_path.write_text(
        "\n".join(
            [
                "route: B_joint_av",
                "scene: scene1_opera",
                "paths:",
                "  manifest: /data/scene_manifest.json",
                "  ftgspp_checkpoint: /data/gaussians.pt",
                "  ftgspp_memmap: /data/memmap/scene1_opera",
                "  output_checkpoint: /out/joint_initialized.pt",
                "train:",
                "  warmup_steps: 7",
                "  joint_steps: 9",
                "  top_k: 5",
                "  audio_lr: 0.003",
                "  shared_lr: 0.0004",
                "  audio_window_seconds: 0.5",
                "  audio_loss_type: audiogs_mono_diff",
                "losses:",
                "  audio_diff_weight: 2.0",
                "  audio_use_log_mag_loss: false",
                "  audio_lre_loss_weight: 0.0",
                "  audio_phase_loss_weight: 0.05",
                "  audio_mr_stft_scales:",
                "    - [256, 64, 256]",
                "    - [512, 160, 400]",
                "  audio_bandpass:",
                "    enable: true",
                "    low_hz: 150.0",
                "    high_hz: -1.0",
                "    order: 5",
            ]
        )
        + "\n"
    )

    args = build_arg_parser().parse_args(["--config", str(config_path)])
    cfg = resolve_training_config(args)

    assert cfg.manifest == "/data/scene_manifest.json"
    assert cfg.ftgspp_checkpoint == "/data/gaussians.pt"
    assert cfg.ftgspp_memmap == "/data/memmap/scene1_opera"
    assert cfg.output == "/out/joint_initialized.pt"
    assert cfg.warmup_steps == 7
    assert cfg.joint_steps == 9
    assert cfg.top_k == 5
    assert cfg.audio_lr == pytest.approx(0.003)
    assert cfg.shared_lr == pytest.approx(0.0004)
    assert cfg.audio_window_seconds == pytest.approx(0.5)
    assert cfg.audio_loss_type == "audiogs_mono_diff"
    assert cfg.audio_diff_weight == pytest.approx(2.0)
    assert cfg.audio_use_log_mag_loss is False
    assert cfg.audio_lre_loss_weight == pytest.approx(0.0)
    assert cfg.audio_phase_loss_weight == pytest.approx(0.05)
    assert cfg.audio_mr_stft_scales == ((256, 64, 256), (512, 160, 400))
    assert cfg.audio_bandpass == {
        "enable": True,
        "low_hz": 150.0,
        "high_hz": -1.0,
        "order": 5,
    }


def test_explicit_cli_args_override_config_file(tmp_path):
    config_path = tmp_path / "route_b.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  manifest: /config/manifest.json",
                "  ftgspp_checkpoint: /config/gaussians.pt",
                "  output_checkpoint: /config/joint.pt",
                "train:",
                "  warmup_steps: 1",
                "  joint_steps: 2",
                "  top_k: 3",
                "  audio_lr: 0.001",
                "  shared_lr: 0.0002",
            ]
        )
        + "\n"
    )

    args = build_arg_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--manifest",
            "/cli/manifest.json",
            "--ftgspp-checkpoint",
            "/cli/gaussians.pt",
            "--output",
            "/cli/joint.pt",
            "--joint-steps",
            "11",
        ]
    )
    cfg = resolve_training_config(args)

    assert cfg.manifest == "/cli/manifest.json"
    assert cfg.ftgspp_checkpoint == "/cli/gaussians.pt"
    assert cfg.output == "/cli/joint.pt"
    assert cfg.warmup_steps == 1
    assert cfg.joint_steps == 11
    assert cfg.top_k == 3


def test_config_file_requires_paths_section(tmp_path):
    config_path = tmp_path / "route_b.yaml"
    config_path.write_text("train:\n  top_k: 4\n")

    args = build_arg_parser().parse_args(["--config", str(config_path)])

    with pytest.raises(ValueError, match="paths.manifest"):
        resolve_training_config(args)


def test_config_rejects_joint_training_without_visual_loss(tmp_path):
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
                "  top_k: 4",
                "  audio_lr: 0.001",
                "  shared_lr: 0.0002",
                "losses:",
                "  visual_weight: 0.0",
            ]
        )
        + "\n"
    )

    args = build_arg_parser().parse_args(["--config", str(config_path)])

    with pytest.raises(ValueError, match="visual_weight.*joint"):
        resolve_training_config(args)


def test_save_joint_checkpoint_writes_route_b_schema(tmp_path):
    model = build_model_from_fake(top_k=4)
    output = tmp_path / "nested" / "joint.pt"
    config = {
        "manifest": "manifest.json",
        "ftgspp_checkpoint": "stage1.pt",
        "output": str(output),
        "warmup_steps": 1,
        "joint_steps": 2,
        "top_k": 4,
        "audio_lr": 5e-4,
        "shared_lr": 1e-5,
    }

    save_joint_checkpoint(
        output_path=output,
        model=model,
        stage="initialized",
        ftgspp_checkpoint="stage1.pt",
        manifest_path="manifest.json",
        config=config,
        loss_history=[0.5, 0.25],
    )

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["route"] == "B_joint_av"
    assert checkpoint["stage"] == "initialized"
    assert checkpoint["ftgspp_checkpoint"] == "stage1.pt"
    assert checkpoint["manifest_path"] == "manifest.json"
    assert "means" in checkpoint["shared_gaussians"]
    assert checkpoint["audio_head"]["mono_gain"].shape == (4, 1)
    assert checkpoint["config"] == config
    assert checkpoint["loss_history"] == [0.5, 0.25]


def test_train_audio_warmup_updates_audio_head_and_records_losses(tmp_path):
    manifest = _write_manifest(tmp_path)
    model = build_model_from_fake(top_k=4)

    before = model.audio_head.mono_gain.detach().clone()
    loss_history = train_audio_warmup(
        model=model,
        manifest_path=manifest,
        steps=2,
        lr=1e-3,
    )

    assert len(loss_history) == 2
    assert all(loss > 0 for loss in loss_history)
    assert not torch.equal(model.audio_head.mono_gain.detach(), before)


def test_train_and_save_writes_audio_warmup_checkpoint(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
    checkpoint_path = tmp_path / "ftgs.pt"
    output_path = tmp_path / "joint_trained.pt"

    def fake_load_checkpoint(path):
        assert path == checkpoint_path
        from avfusion.joint.ftgspp_bridge import FTGSRendererBridge

        return FTGSRendererBridge(FakeGaussians())

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        fake_load_checkpoint,
    )

    summary = train_and_save(
        manifest_path=manifest,
        ftgspp_checkpoint=checkpoint_path,
        output_path=output_path,
        warmup_steps=2,
        joint_steps=0,
        top_k=4,
        audio_lr=1e-3,
        shared_lr=1e-5,
        config_path=None,
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    assert summary["stage"] == "audio_warmup"
    assert summary["steps"] == 2
    assert checkpoint["stage"] == "audio_warmup"
    assert len(checkpoint["loss_history"]) == 2


def test_train_joint_finetune_updates_shared_geometry_and_audio_head(tmp_path):
    manifest = _write_manifest(tmp_path)
    aligned = tmp_path / "audio" / "aligned_16k_stereo"
    _write_wav(aligned / "near.wav", frames=16000, value=0.5)
    _write_wav(aligned / "cam00.wav", frames=16000, value=0.25)
    visual_root = tmp_path / "visual"
    import numpy as np

    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3), np.eye(3)]),
        w2c=np.stack([np.eye(4), np.eye(4)]),
    )
    model = build_model_from_fake(top_k=4)
    with torch.no_grad():
        model.shared_gaussians.means.add_(torch.linspace(0.1, 0.4, 4).reshape(4, 1))
    train_audio_warmup(model=model, manifest_path=manifest, steps=1, lr=1e-3)
    means_before = model.shared_gaussians.means.detach().clone()
    audio_before = model.audio_head.mono_gain.detach().clone()
    render_times = []
    render_w2cs = []
    original_render_audio = model.render_audio

    def spy_render_audio(t, source_audio, camera_w2c=None):
        render_times.append(float(t.detach().cpu().reshape(-1)[0]))
        render_w2cs.append(camera_w2c.detach().cpu().clone() if camera_w2c is not None else None)
        return original_render_audio(t, source_audio, camera_w2c=camera_w2c)

    model.render_audio = spy_render_audio

    losses = train_joint_finetune(
        model=model,
        manifest_path=manifest,
        steps=2,
        shared_lr=1e-3,
        audio_lr=1e-3,
        geometry_reg_weight=0.001,
        rgb_loss_weight=1.0,
        audio_loss_weight=1.0,
        visual_scale=0.5,
        audio_window_seconds=0.5,
        frame_reader=lambda path, frame_idx: torch.ones(4, 4, 3),
    )

    assert len(losses) == 2
    assert all("total" in row and "rgb" in row and "audio" in row and "geo" in row for row in losses)
    assert all("audio_grad_norm" in row and "shared_grad_norm" in row for row in losses)
    assert losses[0]["camera"] == "cam00"
    assert losses[0]["frame"] == 8
    assert losses[0]["time"] == pytest.approx(8 / 30)
    assert losses[0]["start_sample"] == 267
    assert losses[1]["frame"] == 9
    assert render_times[-2:] == pytest.approx([8 / 30, 9 / 30])
    assert render_w2cs[-2:] and all(w2c is not None for w2c in render_w2cs[-2:])
    assert render_w2cs[-1].shape == (1, 4, 4)
    assert not torch.equal(model.shared_gaussians.means.detach(), means_before)
    assert not torch.equal(model.audio_head.mono_gain.detach(), audio_before)


def test_train_joint_finetune_rejects_zero_visual_weight(tmp_path):
    manifest = _write_manifest(tmp_path)
    model = build_model_from_fake(top_k=4)

    with pytest.raises(ValueError, match="rgb_loss_weight"):
        train_joint_finetune(
            model=model,
            manifest_path=manifest,
            steps=1,
            shared_lr=1e-3,
            audio_lr=1e-3,
            geometry_reg_weight=0.001,
            rgb_loss_weight=0.0,
            audio_loss_weight=1.0,
            visual_scale=0.5,
        )


def test_train_and_save_writes_joint_finetune_checkpoint(monkeypatch, tmp_path):
    manifest = _write_manifest(tmp_path)
    aligned = tmp_path / "audio" / "aligned_16k_stereo"
    _write_wav(aligned / "near.wav", frames=16000, value=0.5)
    _write_wav(aligned / "cam00.wav", frames=16000, value=0.25)
    visual_root = tmp_path / "visual"
    import numpy as np

    np.savez(
        visual_root / "cameras.npz",
        names=np.array(["cam00", "cam10"]),
        intrinsics=np.stack([np.eye(3), np.eye(3)]),
        w2c=np.stack([np.eye(4), np.eye(4)]),
    )
    checkpoint_path = tmp_path / "ftgs.pt"
    output_path = tmp_path / "joint_trained.pt"

    def fake_load_checkpoint(path):
        assert path == checkpoint_path
        from avfusion.joint.ftgspp_bridge import FTGSRendererBridge

        return FTGSRendererBridge(FakeGaussians())

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        fake_load_checkpoint,
    )

    summary = train_and_save(
        manifest_path=manifest,
        ftgspp_checkpoint=checkpoint_path,
        output_path=output_path,
        warmup_steps=1,
        joint_steps=2,
        top_k=4,
        audio_lr=1e-3,
        shared_lr=1e-3,
        geometry_reg_weight=0.001,
        rgb_loss_weight=1.0,
        audio_loss_weight=1.0,
        visual_scale=0.5,
        audio_window_seconds=0.5,
        audio_loss_type="audiogs_mono_diff",
        audio_diff_weight=2.0,
        audio_use_log_mag_loss=False,
        audio_lre_loss_weight=0.0,
        audio_bandpass={"enable": True, "low_hz": 150.0, "high_hz": -1.0},
        frame_reader=lambda path, frame_idx: torch.ones(4, 4, 3),
        config_path=None,
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    train_summary = json.loads((tmp_path / "train_summary.json").read_text())
    assert summary["stage"] == "joint_finetune"
    assert summary["steps"] == 3
    assert summary["joint_steps"] == 2
    assert train_summary["train_cams"] == 1
    assert train_summary["eval_cam"] == "cam10"
    assert train_summary["source_audio"] == "near.wav"
    assert train_summary["joint_steps"] == 2
    assert train_summary["audio_window_seconds"] == 0.5
    assert summary["audio_loss_type"] == "audiogs_mono_diff"
    assert checkpoint["stage"] == "joint_finetune"
    assert len(checkpoint["loss_history"]) == 3
    assert checkpoint["config"]["implemented_stages"] == ["audio_warmup", "joint_finetune"]
    assert checkpoint["config"]["audio_loss_type"] == "audiogs_mono_diff"
    assert checkpoint["config"]["audio_bandpass"]["low_hz"] == pytest.approx(150.0)


def test_build_model_clamps_top_k_to_available_gaussians(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "ftgs.pt"

    def fake_load_checkpoint(path):
        assert path == checkpoint_path
        from avfusion.joint.ftgspp_bridge import FTGSRendererBridge

        return FTGSRendererBridge(FakeGaussians())

    monkeypatch.setattr(
        "avfusion.train.train_joint_av_gaussians.FTGSRendererBridge.load_checkpoint",
        fake_load_checkpoint,
    )

    model = build_model(checkpoint_path, top_k=8192)

    assert model.shared_gaussians.means.shape[0] == 4
    assert model.audio_head.active_count == 4


def test_build_model_accepts_checkpoint_payload_with_gaussians_key(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "ftgs_payload.pt"
    torch.save({"gaussians": FakeGaussians()}, checkpoint_path)
    monkeypatch.setattr(
        "avfusion.joint.ftgspp_bridge.importlib.import_module",
        lambda name: object(),
    )

    model = build_model(checkpoint_path, top_k=3)

    assert model.shared_gaussians.means.shape[0] == 4
    assert model.audio_head.active_count == 3


def test_parser_rejects_impossible_step_counts_and_top_k():
    parser = build_arg_parser()

    for flag, value in [
        ("--warmup-steps", "-1"),
        ("--joint-steps", "-1"),
        ("--top-k", "1"),
    ]:
        argv = [
            "--manifest",
            "manifest.json",
            "--ftgspp-checkpoint",
            "stage1.pt",
            "--output",
            "joint.pt",
            flag,
            value,
        ]
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def build_model_from_fake(top_k):
    from avfusion.joint.audio_head import JointAudioHead
    from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
    from avfusion.joint.model import JointAVGaussianModel

    return JointAVGaussianModel(FTGSRendererBridge(FakeGaussians()), JointAudioHead(4, top_k=top_k))
