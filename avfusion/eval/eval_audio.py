from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from avfusion.adapters.visual_to_acoustic import select_topk_acoustic_carrier
from avfusion.audio import AcousticGaussianParameters, render_audio, stft_magnitude_loss
from avfusion.data.audio_video_dataset import AudioCropDataset
from avfusion.visual.carrier import FrozenVisualCarrier


def write_eval_summary(
    output_path: str | Path,
    camera: str,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | str]:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
    if pred.numel() == 0:
        raise ValueError("pred and target must be non-empty")
    if not torch.is_floating_point(pred) or not torch.is_floating_point(target):
        raise ValueError("pred and target must be floating-point tensors")

    l1_waveform = torch.mean(torch.abs(pred - target)).detach().cpu().item()
    summary = {"camera": camera, "l1_waveform": float(l1_waveform)}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate stage-2 audio output.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _load_params(checkpoint: dict) -> AcousticGaussianParameters:
    params_state = checkpoint["params"]
    num_points = int(params_state["mono_gain"].shape[0])
    params = AcousticGaussianParameters(num_points=num_points)
    with torch.no_grad():
        params.mono_gain.copy_(params_state["mono_gain"])
        params.diff_gain.copy_(params_state["diff_gain"])
    return params


def evaluate_audio_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, float | str]:
    dataset = AudioCropDataset(manifest_path, split="eval")
    if len(dataset) != 1:
        raise ValueError(f"expected one eval camera, got {len(dataset)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    carrier = FrozenVisualCarrier.load(checkpoint["carrier_path"])
    acoustic_carrier = select_topk_acoustic_carrier(
        carrier, t=0.0, top_k=int(checkpoint["top_k"])
    )
    params = _load_params(checkpoint)
    sample = dataset[0]

    with torch.no_grad():
        pred = render_audio(acoustic_carrier, params, sample["source_audio"])
        target = sample["target_audio"].to(pred)
        summary = write_eval_summary(
            Path(output_dir) / "audio_summary.json",
            camera=str(sample["camera"]),
            pred=pred,
            target=target,
        )
        summary["stft_magnitude"] = float(
            stft_magnitude_loss(pred, target).detach().cpu().item()
        )
        summary["checkpoint"] = str(checkpoint_path)

    output_path = Path(output_dir) / "audio_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_audio_checkpoint(args.manifest, args.checkpoint, args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
