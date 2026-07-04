from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.model import JointAVGaussianModel


@dataclass(frozen=True)
class TrainingConfig:
    manifest: str
    ftgspp_checkpoint: str
    output: str
    warmup_steps: int
    joint_steps: int
    top_k: int
    audio_lr: float
    shared_lr: float
    config: str | None = None


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be nonnegative, got {parsed}")
    return parsed


def _top_k(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError(f"must be at least 2, got {parsed}")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize Route B joint AV Gaussian fine-tuning from an FTGS++ checkpoint."
    )
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument("--ftgspp-checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--warmup-steps", type=_nonnegative_int)
    parser.add_argument("--joint-steps", type=_nonnegative_int)
    parser.add_argument("--top-k", type=_top_k)
    parser.add_argument("--audio-lr", type=float)
    parser.add_argument("--shared-lr", type=float)
    return parser


def _load_yaml_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    loaded = yaml.safe_load(Path(config_path).read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Route B config must be a mapping, got {type(loaded).__name__}")
    return loaded


def _config_path_value(config: dict[str, Any], key: str) -> str | None:
    paths = config.get("paths") or {}
    if not isinstance(paths, dict):
        raise ValueError("Route B config field paths must be a mapping")
    value = paths.get(key)
    if value is None:
        return None
    return str(value)


def _config_train_value(config: dict[str, Any], key: str, default: Any) -> Any:
    train = config.get("train") or {}
    if not isinstance(train, dict):
        raise ValueError("Route B config field train must be a mapping")
    return train.get(key, default)


def _require_value(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"missing required Route B setting: {name}")
    return value


def resolve_training_config(args: argparse.Namespace) -> TrainingConfig:
    config = _load_yaml_config(args.config)

    manifest = args.manifest or _config_path_value(config, "manifest")
    ftgspp_checkpoint = args.ftgspp_checkpoint or _config_path_value(config, "ftgspp_checkpoint")
    output = args.output or _config_path_value(config, "output_checkpoint")
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = _config_train_value(config, "warmup_steps", 1000)
    joint_steps = args.joint_steps
    if joint_steps is None:
        joint_steps = _config_train_value(config, "joint_steps", 1000)
    top_k = args.top_k
    if top_k is None:
        top_k = _config_train_value(config, "top_k", 8192)
    audio_lr = args.audio_lr
    if audio_lr is None:
        audio_lr = _config_train_value(config, "audio_lr", 5e-4)
    shared_lr = args.shared_lr
    if shared_lr is None:
        shared_lr = _config_train_value(config, "shared_lr", 1e-5)

    warmup_steps = int(_require_value(warmup_steps, "train.warmup_steps"))
    joint_steps = int(_require_value(joint_steps, "train.joint_steps"))
    top_k = int(_require_value(top_k, "train.top_k"))
    audio_lr = float(_require_value(audio_lr, "train.audio_lr"))
    shared_lr = float(_require_value(shared_lr, "train.shared_lr"))
    if warmup_steps < 0:
        raise ValueError(f"train.warmup_steps must be nonnegative, got {warmup_steps}")
    if joint_steps < 0:
        raise ValueError(f"train.joint_steps must be nonnegative, got {joint_steps}")
    if top_k < 2:
        raise ValueError(f"train.top_k must be at least 2, got {top_k}")

    return TrainingConfig(
        manifest=str(_require_value(manifest, "paths.manifest")),
        ftgspp_checkpoint=str(_require_value(ftgspp_checkpoint, "paths.ftgspp_checkpoint")),
        output=str(_require_value(output, "paths.output_checkpoint")),
        warmup_steps=warmup_steps,
        joint_steps=joint_steps,
        top_k=top_k,
        audio_lr=audio_lr,
        shared_lr=shared_lr,
        config=args.config,
    )


def build_model(ftgspp_checkpoint: str | Path, top_k: int = 8192) -> JointAVGaussianModel:
    bridge = FTGSRendererBridge.load_checkpoint(ftgspp_checkpoint)
    means = bridge.gaussians.means
    num_points = int(means.shape[0])
    if num_points < 2:
        raise ValueError(f"Route B joint training requires at least two gaussians, got {num_points}")
    effective_top_k = min(int(top_k), num_points)
    audio_head = JointAudioHead(num_points=num_points, top_k=effective_top_k)
    return JointAVGaussianModel(bridge, audio_head)


def save_joint_checkpoint(
    output_path: str | Path,
    model: JointAVGaussianModel,
    stage: str,
    ftgspp_checkpoint: str | Path,
    manifest_path: str | Path,
    config: dict[str, Any],
    loss_history: list[float],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": "B_joint_av",
            "stage": stage,
            "ftgspp_checkpoint": str(ftgspp_checkpoint),
            "manifest_path": str(manifest_path),
            "shared_gaussians": model.shared_gaussians.state_dict(),
            "audio_head": model.audio_head.state_dict(),
            "config": dict(config),
            "loss_history": list(loss_history),
        },
        output_path,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = resolve_training_config(args)
    model = build_model(config.ftgspp_checkpoint, top_k=config.top_k)
    save_joint_checkpoint(
        output_path=config.output,
        model=model,
        stage="initialized",
        ftgspp_checkpoint=config.ftgspp_checkpoint,
        manifest_path=config.manifest,
        config=asdict(config),
        loss_history=[],
    )
    print(
        {
            "route": "B_joint_av",
            "stage": "initialized",
            "output": config.output,
        }
    )


if __name__ == "__main__":
    main()
