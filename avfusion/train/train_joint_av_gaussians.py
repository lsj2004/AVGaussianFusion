from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from avfusion.joint.audio_head import JointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.model import JointAVGaussianModel


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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ftgspp-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=1000)
    parser.add_argument("--joint-steps", type=_nonnegative_int, default=1000)
    parser.add_argument("--top-k", type=_top_k, default=8192)
    parser.add_argument("--audio-lr", type=float, default=5e-4)
    parser.add_argument("--shared-lr", type=float, default=1e-5)
    return parser


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
    model = build_model(args.ftgspp_checkpoint, top_k=args.top_k)
    save_joint_checkpoint(
        output_path=args.output,
        model=model,
        stage="initialized",
        ftgspp_checkpoint=args.ftgspp_checkpoint,
        manifest_path=args.manifest,
        config=vars(args),
        loss_history=[],
    )
    print(
        {
            "route": "B_joint_av",
            "stage": "initialized",
            "output": args.output,
        }
    )


if __name__ == "__main__":
    main()
