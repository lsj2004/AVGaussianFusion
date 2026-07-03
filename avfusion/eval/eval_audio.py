from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch


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


def main(argv: Sequence[str] | None = None) -> None:
    build_arg_parser().parse_args(argv)
    raise SystemExit(
        "Audio evaluation placeholder: persistent Stage 2 checkpoints are not "
        "implemented yet."
    )


if __name__ == "__main__":
    main()
