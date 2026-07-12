from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from avfusion.data.audio_video_dataset import TimedAudioCropper
from avfusion.data.visual_frame_dataset import FrameReader, VisualFrameDataset
from avfusion.eval.audio_metrics import compute_audiogs_metrics
from avfusion.eval.eval_soft_audio import _restore_soft_model

_FULLTRACK_PROTOCOLS = {
    "audiogs_3s_nonoverlap_fulltrack",
    "visual_center_overlap_add_fulltrack",
}


def _lr_db(audio: torch.Tensor) -> float:
    eps = audio.new_tensor(1e-5)
    value = 10.0 * torch.log10((audio[0].square().sum() + eps) / (audio[1].square().sum() + eps))
    return float(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).item())


def _mid_side_summary(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_mid = 0.5 * (pred[0] + pred[1])
    target_mid = 0.5 * (target[0] + target[1])
    pred_side = 0.5 * (pred[0] - pred[1])
    target_side = 0.5 * (target[0] - target[1])
    eps = pred.new_tensor(1e-5)
    pred_side_mid_db = 10.0 * torch.log10((pred_side.square().sum() + eps) / (pred_mid.square().sum() + eps))
    target_side_mid_db = 10.0 * torch.log10((target_side.square().sum() + eps) / (target_mid.square().sum() + eps))
    return {
        "mid_l1": float(torch.nn.functional.l1_loss(pred_mid, target_mid).item()),
        "side_l1": float(torch.nn.functional.l1_loss(pred_side, target_side).item()),
        "pred_side_mid_db": float(torch.nan_to_num(pred_side_mid_db, nan=0.0, posinf=0.0, neginf=0.0).item()),
        "target_side_mid_db": float(torch.nan_to_num(target_side_mid_db, nan=0.0, posinf=0.0, neginf=0.0).item()),
        "side_mid_db_error": float(torch.abs(pred_side_mid_db - target_side_mid_db).item()),
    }


def _prepare_model(checkpoint_path: str | Path) -> tuple[dict[str, Any], torch.device, Any, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("route") != "C_soft_av_gaussians":
        raise ValueError(f"expected Route C soft checkpoint, got {checkpoint.get('route')!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acoustic_field, audio_renderer = _restore_soft_model(checkpoint)
    acoustic_field.to(device)
    audio_renderer.to(device)
    return checkpoint, device, acoustic_field, audio_renderer


def _render_window(
    acoustic_field,
    audio_renderer,
    source_audio: torch.Tensor,
    camera_w2c: torch.Tensor,
    render_seconds: float,
    device: torch.device,
) -> torch.Tensor:
    render_time = torch.tensor([[render_seconds]], device=device, dtype=torch.float32)
    acoustic_state = acoustic_field.query(render_time)
    return audio_renderer(
        acoustic_state,
        source_audio.to(device),
        camera_w2c=camera_w2c.to(device),
    ).detach().cpu()


def _select_nearest_visual_sample(visual_dataset: VisualFrameDataset, time_seconds: float) -> dict[str, Any]:
    best_idx = min(
        range(len(visual_dataset)),
        key=lambda idx: abs(float(visual_dataset[idx]["time"].reshape(-1)[0].item()) - float(time_seconds)),
    )
    return visual_dataset[best_idx]


def _collect_nonoverlap_windows(
    visual_dataset: VisualFrameDataset,
    cropper: TimedAudioCropper,
    max_windows: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    windows = []
    next_start_seconds = 0.0
    max_visual_time = float(visual_dataset[len(visual_dataset) - 1]["time"].reshape(-1)[0].item())
    while next_start_seconds <= max_visual_time + 1e-6:
        if max_windows is not None and len(windows) >= int(max_windows):
            break
        visual_sample = _select_nearest_visual_sample(visual_dataset, next_start_seconds)
        try:
            audio_sample = cropper.get_crop(str(visual_sample["camera"]), time_seconds=next_start_seconds)
        except ValueError as error:
            if "padding" in str(error):
                break
            raise
        windows.append((visual_sample, audio_sample, float(next_start_seconds)))
        next_start_seconds += cropper.crop_seconds
    return windows


def _collect_visual_center_windows(
    visual_dataset: VisualFrameDataset,
    cropper: TimedAudioCropper,
    max_windows: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    limit = len(visual_dataset) if max_windows is None else min(int(max_windows), len(visual_dataset))
    windows = []
    for idx in range(limit):
        visual_sample = visual_dataset[idx]
        render_seconds = float(visual_sample["time"].reshape(-1)[0].item())
        try:
            audio_sample = cropper.get_crop(str(visual_sample["camera"]), time_seconds=render_seconds)
        except ValueError as error:
            if "padding" in str(error):
                continue
            raise
        windows.append((visual_sample, audio_sample, render_seconds))
    return windows


def _concatenate_windows(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(preds, dim=-1), torch.cat(targets, dim=-1)


def _overlap_add_windows(
    windows: list[tuple[torch.Tensor, torch.Tensor, int]], crop_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    min_start = min(start for _, _, start in windows)
    max_end = max(start + crop_samples for _, _, start in windows)
    total_samples = max_end - min_start
    pred_full = torch.zeros(2, total_samples, dtype=windows[0][0].dtype)
    target_full = torch.zeros(2, total_samples, dtype=windows[0][1].dtype)
    weights = torch.zeros(total_samples, dtype=windows[0][0].dtype)
    window = torch.hann_window(crop_samples, periodic=False, dtype=windows[0][0].dtype).clamp_min(1e-4)
    for pred, target, start_sample in windows:
        offset = start_sample - min_start
        end = offset + crop_samples
        pred_full[:, offset:end] += pred * window.reshape(1, -1)
        target_full[:, offset:end] += target * window.reshape(1, -1)
        weights[offset:end] += window
    weights = weights.clamp_min(1e-8).reshape(1, -1)
    return pred_full / weights, target_full / weights


def evaluate_soft_audio_fulltrack_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    protocol: str = "audiogs_3s_nonoverlap_fulltrack",
    scale: float | None = None,
    ftgspp_memmap: str | Path | None = None,
    audio_window_seconds: float | None = None,
    max_windows: int | None = None,
    frame_reader: FrameReader | None = None,
    include_dpam: bool = False,
    audio_bandpass: dict | None = None,
) -> dict[str, Any]:
    if protocol not in _FULLTRACK_PROTOCOLS:
        raise ValueError(f"unknown fulltrack protocol {protocol!r}")
    checkpoint, device, acoustic_field, audio_renderer = _prepare_model(checkpoint_path)
    config = checkpoint.get("config") or {}
    visual_scale = float(scale if scale is not None else config.get("visual_scale", 0.125))
    memmap_root = ftgspp_memmap if ftgspp_memmap is not None else config.get("ftgspp_memmap")
    if audio_window_seconds is None:
        audio_window_seconds = 3.0 if protocol == "audiogs_3s_nonoverlap_fulltrack" else config.get("audio_window_seconds", 0.5)
    window_seconds = float(audio_window_seconds)
    bandpass = audio_bandpass if audio_bandpass is not None else config.get("audio_bandpass")
    visual_dataset = VisualFrameDataset(
        manifest_path,
        split="eval",
        scale=visual_scale,
        memmap_root=memmap_root,
        frame_reader=frame_reader,
    )
    if len(visual_dataset) == 0:
        raise ValueError("visual eval split is empty")
    cropper = TimedAudioCropper(
        manifest_path,
        crop_seconds=window_seconds,
        mode="start" if protocol == "audiogs_3s_nonoverlap_fulltrack" else "center",
        allow_padding=False,
        bandpass=bandpass,
    )
    if protocol == "audiogs_3s_nonoverlap_fulltrack":
        eval_items = _collect_nonoverlap_windows(visual_dataset, cropper, max_windows)
    else:
        eval_items = _collect_visual_center_windows(visual_dataset, cropper, max_windows)
    if not eval_items:
        raise ValueError(f"no fulltrack windows selected for protocol {protocol!r}")

    preds = []
    targets = []
    ola_items = []
    window_records = []
    with torch.no_grad():
        for visual_sample, audio_sample, render_seconds in eval_items:
            pred = _render_window(
                acoustic_field,
                audio_renderer,
                audio_sample["source_audio"],
                visual_sample["w2c"],
                render_seconds,
                device,
            )
            target = audio_sample["target_audio"].detach().cpu()
            preds.append(pred)
            targets.append(target)
            ola_items.append((pred, target, int(audio_sample["start_sample"])))
            window_records.append(
                {
                    "camera": str(audio_sample["camera"]),
                    "frame": int(visual_sample["frame"]),
                    "time": float(render_seconds),
                    "start_sample": int(audio_sample["start_sample"]),
                }
            )
    if protocol == "audiogs_3s_nonoverlap_fulltrack":
        pred_full, target_full = _concatenate_windows(preds, targets)
        start_sample = int(window_records[0]["start_sample"])
        end_sample = start_sample + int(pred_full.shape[-1])
    else:
        pred_full, target_full = _overlap_add_windows(ola_items, cropper.crop_samples)
        start_sample = min(int(record["start_sample"]) for record in window_records)
        end_sample = max(int(record["start_sample"]) + cropper.crop_samples for record in window_records)

    metrics = compute_audiogs_metrics(
        pred_full,
        target_full,
        sample_rate=int(cropper.sample_rate),
        include_dpam=include_dpam,
    )
    pred_lr_db = _lr_db(pred_full)
    target_lr_db = _lr_db(target_full)
    summary: dict[str, Any] = {
        **metrics,
        **_mid_side_summary(pred_full, target_full),
        "route": "C_soft_av_gaussians",
        "eval_type": "fulltrack",
        "protocol": protocol,
        "checkpoint": str(checkpoint_path),
        "stage": str(checkpoint.get("stage", "unknown")),
        "camera": str(window_records[0]["camera"]),
        "num_windows": int(len(eval_items)),
        "sample_rate": int(cropper.sample_rate),
        "audio_window_seconds": float(window_seconds),
        "num_samples": int(pred_full.shape[-1]),
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "pred_lr_db": pred_lr_db,
        "target_lr_db": target_lr_db,
        "lr_db_error": abs(pred_lr_db - target_lr_db),
        "window_records": window_records,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "pred_full.wav", pred_full.T.numpy(), int(cropper.sample_rate))
    sf.write(output_dir / "target_full.wav", target_full.T.numpy(), int(cropper.sample_rate))
    (output_dir / "full_audio_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Route C checkpoint as continuous full-track audio.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--protocol",
        choices=sorted(_FULLTRACK_PROTOCOLS),
        default="audiogs_3s_nonoverlap_fulltrack",
    )
    parser.add_argument("--scale", type=float)
    parser.add_argument("--ftgspp-memmap")
    parser.add_argument("--audio-window-seconds", type=float)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--include-dpam", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_soft_audio_fulltrack_checkpoint(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        protocol=args.protocol,
        scale=args.scale,
        ftgspp_memmap=args.ftgspp_memmap,
        audio_window_seconds=args.audio_window_seconds,
        max_windows=args.max_windows,
        include_dpam=args.include_dpam,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
