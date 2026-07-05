from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from avfusion.audio import stft_magnitude_loss
from avfusion.data.audio_video_dataset import TimedAudioCropper
from avfusion.data.visual_frame_dataset import FrameReader, VisualFrameDataset
from avfusion.eval.audio_metrics import compute_audiogs_metrics
from avfusion.eval.eval_audio import write_eval_summary
from avfusion.train.train_joint_av_gaussians import build_model


def _average_window_metrics(window_metrics: list[dict]) -> dict:
    if not window_metrics:
        raise ValueError("window_metrics must be non-empty")
    averaged: dict = {}
    for key in ("MAG", "ENV", "LRE", "DPAM"):
        values = [metrics.get(key) for metrics in window_metrics]
        numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
        averaged[key] = (
            sum(numeric_values) / len(numeric_values)
            if numeric_values and len(numeric_values) == len(values)
            else None
        )
    averaged["RTE"] = None
    averaged["RTE_available"] = all(bool(metrics.get("RTE_available")) for metrics in window_metrics)
    averaged["RTE_error"] = window_metrics[0].get("RTE_error")
    averaged["DPAM_available"] = all(bool(metrics.get("DPAM_available")) for metrics in window_metrics)
    averaged["DPAM_error"] = window_metrics[0].get("DPAM_error")
    return averaged


def _restore_joint_model(checkpoint: dict) -> torch.nn.Module:
    config = checkpoint.get("config") or {}
    top_k = int(config.get("top_k", 8192))
    model = build_model(checkpoint["ftgspp_checkpoint"], top_k=top_k)
    model.shared_gaussians.load_state_dict(checkpoint["shared_gaussians"], strict=False)
    model.audio_head.load_state_dict(checkpoint["audio_head"])
    model.eval()
    return model


def evaluate_joint_audio_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    scale: float | None = None,
    ftgspp_memmap: str | Path | None = None,
    audio_window_seconds: float | None = None,
    max_frames: int | None = None,
    frame_reader: FrameReader | None = None,
) -> dict[str, float | str | dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("route") != "B_joint_av":
        raise ValueError(f"expected Route B joint checkpoint, got {checkpoint.get('route')!r}")
    config = checkpoint.get("config") or {}
    visual_scale = float(scale if scale is not None else config.get("visual_scale", 0.125))
    memmap_root = ftgspp_memmap if ftgspp_memmap is not None else config.get("ftgspp_memmap")
    window_seconds = float(
        audio_window_seconds
        if audio_window_seconds is not None
        else config.get("audio_window_seconds", 0.5)
    )
    visual_dataset = VisualFrameDataset(
        manifest_path,
        split="eval",
        scale=visual_scale,
        memmap_root=memmap_root,
        frame_reader=frame_reader,
    )
    if len(visual_dataset) == 0:
        raise ValueError("visual eval split is empty")
    audio_cropper = TimedAudioCropper(
        manifest_path,
        crop_seconds=window_seconds,
        mode="center",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _restore_joint_model(checkpoint).to(device)
    limit = len(visual_dataset) if max_frames is None else min(int(max_frames), len(visual_dataset))
    if limit <= 0:
        raise ValueError(f"max_frames must be positive when provided, got {max_frames}")
    preds = []
    targets = []
    window_metrics = []
    camera = None

    with torch.no_grad():
        for idx in range(limit):
            visual_sample = visual_dataset[idx]
            camera = str(visual_sample["camera"])
            audio_sample = audio_cropper.get_crop(camera, visual_sample["time"])
            render_time = visual_sample["time"].to(device)
            pred_window = model.render_audio(
                render_time,
                audio_sample["source_audio"].to(device),
            )
            preds.append(pred_window.detach().cpu())
            target_window = audio_sample["target_audio"].to(pred_window)
            targets.append(target_window.detach().cpu())
            metrics = compute_audiogs_metrics(
                pred_window.detach().cpu(),
                target_window.detach().cpu(),
                sample_rate=int(audio_cropper.manifest.audio.sample_rate),
                include_dpam=False,
            )
            metrics["camera"] = str(camera)
            metrics["frame"] = int(visual_sample["frame"])
            metrics["time"] = float(visual_sample["time"].detach().cpu().reshape(-1)[0].item())
            metrics["start_sample"] = int(audio_sample["start_sample"])
            window_metrics.append(metrics)
        pred = torch.cat(preds, dim=-1)
        target = torch.cat(targets, dim=-1)
        debug = write_eval_summary(
            Path(output_dir) / "audio_debug_summary.json",
            camera=str(camera),
            pred=pred,
            target=target,
        )
        debug["stft_magnitude"] = float(
            stft_magnitude_loss(pred, target).detach().cpu().item()
        )
        summary = _average_window_metrics(window_metrics)
        summary["camera"] = str(camera)
        summary["checkpoint"] = str(checkpoint_path)
        summary["stage"] = str(checkpoint.get("stage", "unknown"))
        summary["num_windows"] = int(limit)
        summary["audio_window_seconds"] = float(window_seconds)
        summary["window_metrics"] = window_metrics
        summary["concatenated_overlapping_debug"] = debug

    output_path = Path(output_dir) / "audio_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Route B joint audio output.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--ftgspp-memmap")
    parser.add_argument("--audio-window-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_joint_audio_checkpoint(
        args.manifest,
        args.checkpoint,
        args.output_dir,
        scale=args.scale,
        ftgspp_memmap=args.ftgspp_memmap,
        audio_window_seconds=args.audio_window_seconds,
        max_frames=args.max_frames,
    )
    print(summary)


if __name__ == "__main__":
    main()
