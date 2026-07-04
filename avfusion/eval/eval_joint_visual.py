from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch

from avfusion.data.visual_frame_dataset import FrameReader, VisualFrameDataset
from avfusion.eval.eval_joint_audio import _restore_joint_model
from avfusion.train.train_joint_av_gaussians import _move_tensor_values


def _psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return 100.0
    return float(-10.0 * math.log10(mse))


def evaluate_joint_visual_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    scale: float | None = None,
    ftgspp_memmap: str | Path | None = None,
    max_frames: int | None = None,
    frame_reader: FrameReader | None = None,
) -> dict[str, float | int | str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("route") != "B_joint_av":
        raise ValueError(f"expected Route B joint checkpoint, got {checkpoint.get('route')!r}")
    config = checkpoint.get("config") or {}
    visual_scale = float(scale if scale is not None else config.get("visual_scale", 0.125))
    memmap_root = ftgspp_memmap if ftgspp_memmap is not None else config.get("ftgspp_memmap")
    dataset = VisualFrameDataset(
        manifest_path,
        split="eval",
        scale=visual_scale,
        memmap_root=memmap_root,
        frame_reader=frame_reader,
    )
    if len(dataset) == 0:
        raise ValueError("visual eval split is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _restore_joint_model(checkpoint).to(device)
    limit = len(dataset) if max_frames is None else min(int(max_frames), len(dataset))
    if limit <= 0:
        raise ValueError(f"max_frames must be positive when provided, got {max_frames}")

    total_mse = 0.0
    total_l1 = 0.0
    total_pixels = 0
    camera = None
    with torch.no_grad():
        for idx in range(limit):
            sample = dataset[idx]
            camera = str(sample["camera"])
            batch = _move_tensor_values(sample, device)
            pred = model.render_rgb(batch).clamp(0.0, 1.0)
            target = batch["target_rgb"].to(pred)
            diff = pred - target
            total_mse += float(torch.sum(diff.square()).detach().cpu().item())
            total_l1 += float(torch.sum(diff.abs()).detach().cpu().item())
            total_pixels += int(diff.numel())

    mse = total_mse / total_pixels
    l1 = total_l1 / total_pixels
    summary: dict[str, float | int | str] = {
        "PSNR": _psnr_from_mse(mse),
        "MSE": float(mse),
        "L1": float(l1),
        "num_frames": int(limit),
        "camera": str(camera),
        "checkpoint": str(checkpoint_path),
        "stage": str(checkpoint.get("stage", "unknown")),
        "visual_scale": float(visual_scale),
    }

    output_path = Path(output_dir) / "visual_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Route B joint visual PSNR.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--ftgspp-memmap")
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_joint_visual_checkpoint(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        scale=args.scale,
        ftgspp_memmap=args.ftgspp_memmap,
        max_frames=args.max_frames,
    )
    print(summary)


if __name__ == "__main__":
    main()
