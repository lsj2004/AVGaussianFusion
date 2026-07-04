import pytest
import torch

from avfusion.train.train_joint_av_gaussians import (
    build_arg_parser,
    build_model,
    save_joint_checkpoint,
)
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
