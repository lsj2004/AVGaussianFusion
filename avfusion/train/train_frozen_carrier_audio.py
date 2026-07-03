from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.adapters.visual_to_acoustic import select_topk_acoustic_carrier
from avfusion.audio import AcousticGaussianParameters, render_audio, stft_magnitude_loss
from avfusion.data.audio_video_dataset import AudioCropDataset
from avfusion.visual.carrier import FrozenVisualCarrier


def train_one_step(
    acoustic_carrier: AcousticCarrier,
    source_audio: torch.Tensor,
    target_audio: torch.Tensor,
    lr: float,
    params: AcousticGaussianParameters | None = None,
) -> tuple[float, AcousticGaussianParameters]:
    if params is None:
        params = AcousticGaussianParameters(num_points=acoustic_carrier.xyz.shape[0])
    optimizer = torch.optim.Adam(params.parameters(), lr=lr)

    optimizer.zero_grad(set_to_none=True)
    pred = render_audio(acoustic_carrier, params, source_audio)
    loss = stft_magnitude_loss(pred, target_audio.to(pred))
    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu().item()), params


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train stage-2 audio parameters from a frozen visual carrier."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=8192)
    return parser


def train_audio_parameters(
    manifest_path: str | Path,
    carrier_path: str | Path,
    output_path: str | Path,
    steps: int,
    lr: float,
    top_k: int,
) -> dict[str, float | int | list[str]]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    dataset = AudioCropDataset(manifest_path, split="train")
    if len(dataset) == 0:
        raise ValueError("training split is empty")

    carrier = FrozenVisualCarrier.load(carrier_path)
    acoustic_carrier = select_topk_acoustic_carrier(carrier, t=0.0, top_k=top_k)
    params = AcousticGaussianParameters(num_points=acoustic_carrier.xyz.shape[0])
    optimizer = torch.optim.Adam(params.parameters(), lr=lr)
    loss_history: list[float] = []

    for step in range(steps):
        sample = dataset[step % len(dataset)]
        optimizer.zero_grad(set_to_none=True)
        pred = render_audio(acoustic_carrier, params, sample["source_audio"])
        loss = stft_magnitude_loss(pred, sample["target_audio"].to(pred))
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "manifest_path": str(manifest_path),
            "carrier_path": str(carrier_path),
            "top_k": int(top_k),
            "params": {
                "mono_gain": params.mono_gain.detach().cpu(),
                "diff_gain": params.diff_gain.detach().cpu(),
            },
            "loss_history": loss_history,
            "train_cameras": list(dataset.camera_names),
        },
        output_path,
    )

    return {
        "steps": int(steps),
        "final_loss": loss_history[-1],
        "top_k": int(top_k),
        "train_cameras": list(dataset.camera_names),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = train_audio_parameters(
        manifest_path=args.manifest,
        carrier_path=args.carrier,
        output_path=args.output,
        steps=args.steps,
        lr=args.lr,
        top_k=args.top_k,
    )
    print(summary)


if __name__ == "__main__":
    main()
