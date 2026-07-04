from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from avfusion.audio import stft_magnitude_loss
from avfusion.data.audio_video_dataset import AudioCropDataset
from avfusion.eval.audio_metrics import compute_audiogs_metrics
from avfusion.eval.eval_audio import write_eval_summary
from avfusion.train.train_joint_av_gaussians import build_model


def _restore_joint_model(checkpoint: dict) -> torch.nn.Module:
    config = checkpoint.get("config") or {}
    top_k = int(config.get("top_k", 8192))
    model = build_model(checkpoint["ftgspp_checkpoint"], top_k=top_k)
    model.shared_gaussians.load_state_dict(checkpoint["shared_gaussians"], strict=False)
    model.audio_head.load_state_dict(checkpoint["audio_head"])
    model.eval()
    return model


def evaluate_joint_audio_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, float | str | dict]:
    dataset = AudioCropDataset(manifest_path, split="eval")
    if len(dataset) != 1:
        raise ValueError(f"expected one eval camera, got {len(dataset)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("route") != "B_joint_av":
        raise ValueError(f"expected Route B joint checkpoint, got {checkpoint.get('route')!r}")
    model = _restore_joint_model(checkpoint)
    sample = dataset[0]

    with torch.no_grad():
        pred = model.render_audio(torch.tensor([[0.0]]), sample["source_audio"])
        target = sample["target_audio"].to(pred)
        debug = write_eval_summary(
            Path(output_dir) / "audio_debug_summary.json",
            camera=str(sample["camera"]),
            pred=pred,
            target=target,
        )
        debug["stft_magnitude"] = float(
            stft_magnitude_loss(pred, target).detach().cpu().item()
        )
        summary = compute_audiogs_metrics(
            pred,
            target,
            sample_rate=int(dataset.manifest.audio.sample_rate),
            include_dpam=True,
        )
        summary["camera"] = str(sample["camera"])
        summary["checkpoint"] = str(checkpoint_path)
        summary["stage"] = str(checkpoint.get("stage", "unknown"))
        summary["debug"] = debug

    output_path = Path(output_dir) / "audio_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Route B joint audio output.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_joint_audio_checkpoint(args.manifest, args.checkpoint, args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
