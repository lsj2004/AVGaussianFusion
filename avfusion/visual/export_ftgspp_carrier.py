from __future__ import annotations

import argparse
import sys
import types
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


def _is_tinycudann_compute_capability_error(error: OSError) -> bool:
    return "Unknown compute capability" in str(error)


def _install_tinycudann_unpickle_stub() -> None:
    module = types.ModuleType("tinycudann")
    modules = types.ModuleType("tinycudann.modules")

    class _UnavailableTinyCudaNN:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "tinycudann is unavailable in this CPU export environment"
            )

    def _unsupported(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("tinycudann is unavailable in this CPU export environment")

    module.Encoding = _UnavailableTinyCudaNN
    module.Network = _UnavailableTinyCudaNN
    module.NetworkWithInputEncoding = _UnavailableTinyCudaNN
    module.supports_jit_fusion = lambda: False
    module.free_temporary_memory = _unsupported

    modules.Encoding = module.Encoding
    modules.Network = module.Network
    modules.NetworkWithInputEncoding = module.NetworkWithInputEncoding
    modules.supports_jit_fusion = module.supports_jit_fusion
    modules.free_temporary_memory = module.free_temporary_memory

    sys.modules["tinycudann"] = module
    sys.modules["tinycudann.modules"] = modules


def _load_checkpoint_for_cpu_export(checkpoint_path: str | Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except OSError as error:
        if not _is_tinycudann_compute_capability_error(error):
            raise
        _install_tinycudann_unpickle_stub()
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def export_checkpoint(
    checkpoint_path: str | Path, output_path: str | Path
) -> FrozenVisualCarrier:
    checkpoint = _load_checkpoint_for_cpu_export(checkpoint_path)
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
