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

DEFAULT_DPAM_MAX_WINDOWS = 16


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


def _select_dpam_window_indices(num_windows: int, dpam_max_windows: int | None) -> list[int]:
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")
    if dpam_max_windows is not None and dpam_max_windows <= 0:
        raise ValueError(f"dpam_max_windows must be positive when provided, got {dpam_max_windows}")
    limit = min(num_windows, dpam_max_windows or DEFAULT_DPAM_MAX_WINDOWS)
    return list(range(limit))


def _attach_sampled_dpam_metrics(
    summary: dict,
    dpam_window_metrics: list[dict],
) -> None:
    summary["dpam_num_windows"] = len(dpam_window_metrics)
    summary["dpam_window_metrics"] = dpam_window_metrics
    if not dpam_window_metrics:
        summary["DPAM"] = None
        summary["DPAM_available"] = False
        summary["DPAM_error"] = "no windows selected"
        return

    values = [metrics.get("DPAM") for metrics in dpam_window_metrics]
    numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
    summary["DPAM"] = (
        sum(numeric_values) / len(numeric_values)
        if numeric_values and len(numeric_values) == len(values)
        else None
    )
    summary["DPAM_available"] = all(
        bool(metrics.get("DPAM_available")) for metrics in dpam_window_metrics
    )
    errors = [metrics.get("DPAM_error") for metrics in dpam_window_metrics if metrics.get("DPAM_error")]
    summary["DPAM_error"] = errors[0] if errors else None


def _restore_joint_model(checkpoint: dict) -> torch.nn.Module:
    config = checkpoint.get("config") or {}
    top_k = int(config.get("top_k", 8192))
    audio_head_type = str(config.get("audio_head_type", "simple"))
    model = build_model(
        checkpoint["ftgspp_checkpoint"],
        top_k=top_k,
        audio_head_type=audio_head_type,
    )
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
    include_dpam: bool = False,
    dpam_max_windows: int | None = None,
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
                camera_w2c=visual_sample["w2c"].to(device),
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
        if include_dpam:
            dpam_window_metrics = []
            for window_idx in _select_dpam_window_indices(limit, dpam_max_windows):
                metrics = compute_audiogs_metrics(
                    preds[window_idx],
                    targets[window_idx],
                    sample_rate=int(audio_cropper.manifest.audio.sample_rate),
                    include_dpam=True,
                )
                metrics["source_window_index"] = int(window_idx)
                metrics["camera"] = window_metrics[window_idx]["camera"]
                metrics["frame"] = window_metrics[window_idx]["frame"]
                metrics["time"] = window_metrics[window_idx]["time"]
                metrics["start_sample"] = window_metrics[window_idx]["start_sample"]
                dpam_window_metrics.append(metrics)
            _attach_sampled_dpam_metrics(summary, dpam_window_metrics)
        else:
            summary["dpam_num_windows"] = 0

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
    parser.add_argument("--include-dpam", action="store_true")
    parser.add_argument(
        "--dpam-max-windows",
        type=int,
        default=DEFAULT_DPAM_MAX_WINDOWS,
        help="Maximum sampled windows for DPAM when --include-dpam is set.",
    )
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
        include_dpam=args.include_dpam,
        dpam_max_windows=args.dpam_max_windows,
    )
    print(summary)


if __name__ == "__main__":
    main()
