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
from avfusion.eval.eval_joint_audio import _restore_joint_model
from avfusion.eval.eval_soft_audio_fulltrack import (
    _FULLTRACK_PROTOCOLS,
    _collect_nonoverlap_windows,
    _collect_visual_center_windows,
    _lr_db,
    _mid_side_summary,
    _overlap_add_windows,
)


def _prepare_model(checkpoint_path: str | Path) -> tuple[dict[str, Any], torch.device, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("route") != "B_joint_av":
        raise ValueError(f"expected Route B joint checkpoint, got {checkpoint.get('route')!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _restore_joint_model(checkpoint).to(device)
    return checkpoint, device, model


def _render_window(
    model: torch.nn.Module,
    source_audio: torch.Tensor,
    camera_w2c: torch.Tensor,
    render_seconds: float,
    device: torch.device,
) -> torch.Tensor:
    render_time = torch.tensor([[render_seconds]], device=device, dtype=torch.float32)
    return model.render_audio(
        render_time,
        source_audio.to(device),
        camera_w2c=camera_w2c.to(device),
    ).detach().cpu()


def _concatenate_windows(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(preds, dim=-1), torch.cat(targets, dim=-1)


def evaluate_joint_audio_fulltrack_checkpoint(
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
    checkpoint, device, model = _prepare_model(checkpoint_path)
    config = checkpoint.get("config") or {}
    visual_scale = float(scale if scale is not None else config.get("visual_scale", 0.125))
    memmap_root = ftgspp_memmap if ftgspp_memmap is not None else config.get("ftgspp_memmap")
    if audio_window_seconds is None:
        audio_window_seconds = 3.0 if protocol == "audiogs_3s_nonoverlap_fulltrack" else config.get(
            "audio_window_seconds", 0.5
        )
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
                model,
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
        "route": "B_joint_av",
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
    parser = argparse.ArgumentParser(description="Evaluate a Route B checkpoint as continuous full-track audio.")
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
    summary = evaluate_joint_audio_fulltrack_checkpoint(
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
