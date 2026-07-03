from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from avfusion.visual.carrier import FrozenVisualCarrier


def _extract_velocity(gs: Any) -> torch.Tensor:
    velocity_model = getattr(gs, "velocity_model")
    if isinstance(velocity_model, torch.Tensor):
        return velocity_model
    raise TypeError(
        "velocity_model must be an explicit tensor in the first carrier exporter"
    )


def carrier_from_ftgspp_object(gs: Any) -> FrozenVisualCarrier:
    kwargs = {
        "means": getattr(gs, "means"),
        "scales": getattr(gs, "scales"),
        "quats": getattr(gs, "quats"),
        "opacities": getattr(gs, "opacities"),
        "times": getattr(gs, "times"),
        "durations": getattr(gs, "durations"),
        "velocities": _extract_velocity(gs),
        "max_duration": getattr(gs, "max_duration"),
    }
    if hasattr(gs, "marginal_gates"):
        kwargs["marginal_gates"] = getattr(gs, "marginal_gates")
    return FrozenVisualCarrier(**kwargs)


def export_checkpoint(
    checkpoint_path: str | Path, output_path: str | Path
) -> FrozenVisualCarrier:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "gaussians" in checkpoint:
        gs = checkpoint["gaussians"]
    else:
        gs = checkpoint
    carrier = carrier_from_ftgspp_object(gs)
    carrier.save(output_path)
    return carrier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export FTGS++ gaussians as a frozen visual carrier."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the FTGS++ checkpoint."
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the frozen carrier."
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_checkpoint(args.checkpoint, output_path)


if __name__ == "__main__":
    main()
