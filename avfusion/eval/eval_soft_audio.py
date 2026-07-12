from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from avfusion.data.audio_video_dataset import TimedAudioCropper
from avfusion.data.visual_frame_dataset import FrameReader, VisualFrameDataset
from avfusion.eval.audio_metrics import compute_audiogs_metrics
from avfusion.eval.eval_joint_audio import (
    _attach_sampled_dpam_metrics,
    _average_window_metrics,
    _select_dpam_window_indices,
)
from avfusion.soft.acoustic_field import AcousticGaussianField
from avfusion.soft.frequency_transfer_renderer import FrequencyTransferRenderer


def _restore_soft_model(checkpoint: dict) -> tuple[AcousticGaussianField, FrequencyTransferRenderer]:
    field_state = checkpoint.get("acoustic_field")
    if not isinstance(field_state, dict):
        raise ValueError("Route C checkpoint is missing acoustic_field state")
    field = AcousticGaussianField(
        means=field_state["means"],
        opacities=field_state["opacities"],
        velocity_model=field_state["velocity_model"],
        times=field_state["times"],
        anchor_indices=field_state["anchor_indices"],
        anchor_weights=field_state["anchor_weights"],
        anchor_mask=field_state["anchor_mask"],
        residual_mask=field_state["residual_mask"],
        audio_opacity=field_state["audio_opacity"],
        mono_response=field_state["mono_response"],
        diff_response=field_state["diff_response"],
        distance_decay=field_state["distance_decay"],
        phase_delay=field_state["phase_delay"],
    )
    field.load_state_dict(field_state, strict=False)
    renderer_config = checkpoint.get("audio_renderer") or {}
    renderer = FrequencyTransferRenderer(
        n_fft=int(renderer_config.get("n_fft", 512)),
        hop_length=int(renderer_config.get("hop_length", 160)),
        win_length=int(renderer_config.get("win_length", 400)),
        use_phase_delay=bool(renderer_config.get("use_phase_delay", False)),
    )
    field.eval()
    renderer.eval()
    return field, renderer


def evaluate_soft_audio_checkpoint(
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
    audio_bandpass: dict | None = None,
    audio_eval_protocol: str = "visual_center",
    skip_padding_windows: bool = False,
) -> dict[str, float | str | dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("route") != "C_soft_av_gaussians":
        raise ValueError(f"expected Route C soft checkpoint, got {checkpoint.get('route')!r}")
    config = checkpoint.get("config") or {}
    visual_scale = float(scale if scale is not None else config.get("visual_scale", 0.125))
    memmap_root = ftgspp_memmap if ftgspp_memmap is not None else config.get("ftgspp_memmap")
    window_seconds = float(
        audio_window_seconds
        if audio_window_seconds is not None
        else config.get("audio_window_seconds", 0.5)
    )
    bandpass = audio_bandpass if audio_bandpass is not None else config.get("audio_bandpass")
    if audio_eval_protocol not in {
        "visual_center",
        "visual_center_skip_padding",
        "audiogs_start_nonoverlap",
    }:
        raise ValueError(f"unknown audio_eval_protocol {audio_eval_protocol!r}")
    if skip_padding_windows and audio_eval_protocol == "visual_center":
        audio_eval_protocol = "visual_center_skip_padding"
    crop_mode = "start" if audio_eval_protocol == "audiogs_start_nonoverlap" else "center"
    allow_padding = audio_eval_protocol == "visual_center"
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
        mode=crop_mode,
        allow_padding=allow_padding,
        bandpass=bandpass,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acoustic_field, audio_renderer = _restore_soft_model(checkpoint)
    acoustic_field.to(device)
    audio_renderer.to(device)
    limit = len(visual_dataset) if max_frames is None else min(int(max_frames), len(visual_dataset))
    if limit <= 0:
        raise ValueError(f"max_frames must be positive when provided, got {max_frames}")

    if audio_eval_protocol == "audiogs_start_nonoverlap":
        eval_items = []
        next_start_seconds = 0.0
        max_visual_time = float(visual_dataset[len(visual_dataset) - 1]["time"].reshape(-1)[0].item())
        while len(eval_items) < limit and next_start_seconds <= max_visual_time + 1e-6:
            best_idx = min(
                range(len(visual_dataset)),
                key=lambda idx: abs(
                    float(visual_dataset[idx]["time"].reshape(-1)[0].item()) - next_start_seconds
                ),
            )
            visual_sample = visual_dataset[best_idx]
            try:
                audio_sample = audio_cropper.get_crop(str(visual_sample["camera"]), time_seconds=next_start_seconds)
            except ValueError as error:
                if "padding" in str(error):
                    break
                raise
            eval_items.append((visual_sample, audio_sample, float(next_start_seconds)))
            next_start_seconds += window_seconds
    else:
        eval_items = []
        skipped_padding_windows = 0
        for idx in range(limit):
            visual_sample = visual_dataset[idx]
            try:
                audio_sample = audio_cropper.get_crop(str(visual_sample["camera"]), visual_sample["time"])
            except ValueError as error:
                if audio_eval_protocol == "visual_center_skip_padding" and "padding" in str(error):
                    skipped_padding_windows += 1
                    continue
                raise
            eval_items.append(
                (
                    visual_sample,
                    audio_sample,
                    float(visual_sample["time"].reshape(-1)[0].item()),
                )
            )
    if not eval_items:
        raise ValueError("no valid audio eval windows")

    preds = []
    targets = []
    window_metrics = []
    camera = None
    dpam_indices = set(_select_dpam_window_indices(len(eval_items), dpam_max_windows)) if include_dpam else set()
    dpam_window_metrics = []
    with torch.no_grad():
        for window_idx, (visual_sample, audio_sample, render_seconds) in enumerate(eval_items):
            visual_sample = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in visual_sample.items()
            }
            render_time = torch.tensor([[render_seconds]], device=device, dtype=torch.float32)
            acoustic_state = acoustic_field.query(render_time)
            pred = audio_renderer(
                acoustic_state,
                audio_sample["source_audio"].to(device),
                camera_w2c=visual_sample["w2c"],
            )
            target = audio_sample["target_audio"].to(pred)
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
            camera = str(audio_sample["camera"])
            include_window_dpam = include_dpam and window_idx in dpam_indices
            metrics = compute_audiogs_metrics(
                pred.detach().cpu(),
                target.detach().cpu(),
                sample_rate=audio_cropper.sample_rate,
                include_dpam=include_window_dpam,
            )
            metrics.update(
                {
                    "frame": int(visual_sample["frame"]),
                    "time": float(render_seconds),
                    "start_sample": int(audio_sample["start_sample"]),
                }
            )
            window_metrics.append(metrics)
            if include_window_dpam:
                dpam_window_metrics.append(metrics)

    pred_all = torch.cat(preds, dim=-1)
    target_all = torch.cat(targets, dim=-1)
    debug = {
        "l1_waveform": float(torch.nn.functional.l1_loss(pred_all, target_all).item()),
        "stft_magnitude": float(0.0),
    }
    summary = _average_window_metrics(window_metrics)
    summary.update(
        {
            "route": "C_soft_av_gaussians",
            "renderer_type": checkpoint.get("renderer_type", "frequency_transfer"),
            "checkpoint": str(checkpoint_path),
            "stage": str(checkpoint.get("stage", "unknown")),
            "camera": camera,
            "num_windows": int(len(eval_items)),
            "requested_windows": int(limit),
            "skipped_padding_windows": int(
                skipped_padding_windows if audio_eval_protocol == "visual_center_skip_padding" else 0
            ),
            "sample_rate": audio_cropper.sample_rate,
            "audio_window_seconds": float(window_seconds),
            "audio_eval_protocol": audio_eval_protocol,
            "window_metrics": window_metrics,
            "concatenated_overlapping_debug": debug,
        }
    )
    if include_dpam:
        _attach_sampled_dpam_metrics(summary, dpam_window_metrics)
    else:
        summary["dpam_num_windows"] = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "window_metrics.json").write_text(json.dumps(window_metrics, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Route C soft AV audio checkpoint.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--ftgspp-memmap")
    parser.add_argument("--audio-window-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--include-dpam", action="store_true")
    parser.add_argument("--dpam-max-windows", type=int)
    parser.add_argument(
        "--audio-eval-protocol",
        choices=["visual_center", "visual_center_skip_padding", "audiogs_start_nonoverlap"],
        default="visual_center",
    )
    parser.add_argument("--skip-padding-windows", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_soft_audio_checkpoint(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        scale=args.scale,
        ftgspp_memmap=args.ftgspp_memmap,
        audio_window_seconds=args.audio_window_seconds,
        max_frames=args.max_frames,
        include_dpam=args.include_dpam,
        dpam_max_windows=args.dpam_max_windows,
        audio_eval_protocol=args.audio_eval_protocol,
        skip_padding_windows=args.skip_padding_windows,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
