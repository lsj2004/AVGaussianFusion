from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from avfusion.adapters.visual_to_acoustic import select_topk_acoustic_carrier
from avfusion.audio import AcousticGaussianParameters, render_audio
from avfusion.data.audio_video_dataset import _read_audio_full
from avfusion.data.manifest import SceneManifest
from avfusion.eval.audio_metrics import compute_audiogs_metrics
from avfusion.eval.eval_audio import _load_params
from avfusion.eval.eval_soft_audio_fulltrack import _lr_db, _mid_side_summary
from avfusion.visual.carrier import FrozenVisualCarrier


def evaluate_audio_fulltrack_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    include_dpam: bool = False,
    bandpass: dict[str, float | bool] | None = None,
) -> dict[str, Any]:
    manifest = SceneManifest.load(manifest_path)
    if len(manifest.eval_cameras) != 1:
        raise ValueError(f"expected one eval camera, got {len(manifest.eval_cameras)}")
    camera = manifest.eval_cameras[0]
    record = manifest.cameras[camera]
    sample_rate = int(manifest.audio.sample_rate)
    channels = int(manifest.audio.channels)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    carrier = FrozenVisualCarrier.load(checkpoint["carrier_path"])
    acoustic_carrier = select_topk_acoustic_carrier(
        carrier,
        t=0.0,
        top_k=int(checkpoint["top_k"]),
    )
    params = _load_params(checkpoint)
    source_full = _read_audio_full(
        manifest.audio.source_path,
        sample_rate,
        channels,
        bandpass=bandpass,
    )
    target_full = _read_audio_full(
        record.audio_path,
        sample_rate,
        channels,
        bandpass=bandpass,
    )
    num_samples = min(int(source_full.shape[-1]), int(target_full.shape[-1]))
    source_full = source_full[:, :num_samples]
    target_full = target_full[:, :num_samples]
    with torch.no_grad():
        pred_full = render_audio(acoustic_carrier, params, source_full).detach().cpu()
    metrics = compute_audiogs_metrics(
        pred_full,
        target_full,
        sample_rate=sample_rate,
        include_dpam=include_dpam,
    )
    pred_lr_db = _lr_db(pred_full)
    target_lr_db = _lr_db(target_full)
    summary: dict[str, Any] = {
        **metrics,
        **_mid_side_summary(pred_full, target_full),
        "route": "A_frozen_visual_audio_head",
        "eval_type": "fulltrack",
        "protocol": "source_to_heldout_fulltrack",
        "checkpoint": str(checkpoint_path),
        "camera": camera,
        "num_windows": 1,
        "sample_rate": sample_rate,
        "num_samples": int(num_samples),
        "start_sample": 0,
        "end_sample": int(num_samples),
        "pred_lr_db": pred_lr_db,
        "target_lr_db": target_lr_db,
        "lr_db_error": abs(pred_lr_db - target_lr_db),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "pred_full.wav", pred_full.T.numpy(), sample_rate)
    sf.write(output_dir / "target_full.wav", target_full.T.numpy(), sample_rate)
    (output_dir / "full_audio_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Route A audio head as continuous full-track audio.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-dpam", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_audio_fulltrack_checkpoint(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        include_dpam=args.include_dpam,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
