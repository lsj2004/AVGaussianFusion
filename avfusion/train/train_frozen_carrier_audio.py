from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from avfusion.adapters.visual_to_acoustic import AcousticCarrier
from avfusion.audio import AcousticGaussianParameters, render_audio, stft_magnitude_loss


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
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    build_arg_parser().parse_args(argv)
    raise SystemExit(
        "Stage 2 audio training placeholder: carrier export/checkpoint is not "
        "available yet."
    )


if __name__ == "__main__":
    main()
