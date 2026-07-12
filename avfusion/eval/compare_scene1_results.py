from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from avfusion.adapters.visual_to_acoustic import select_topk_acoustic_carrier
from avfusion.audio import AcousticGaussianParameters, render_audio
from avfusion.data.audio_video_dataset import AudioCropDataset, TimedAudioCropper
from avfusion.data.manifest import SceneManifest
from avfusion.data.visual_frame_dataset import VisualFrameDataset
from avfusion.eval.eval_joint_audio import _restore_joint_model
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.train.train_joint_av_gaussians import _move_tensor_values
from avfusion.visual.carrier import FrozenVisualCarrier


METRIC_COLUMNS = [
    "method",
    "schedule",
    "audio_protocol",
    "visual_protocol",
    "MAG",
    "ENV",
    "LRE",
    "DPAM",
    "DPAM_available",
    "DPAM_error",
    "RTE",
    "RTE_available",
    "PSNR",
    "SSIM-1",
    "LPIPS-Alex",
    "audio_camera",
    "visual_camera",
    "audio_windows",
    "audio_window_seconds",
    "visual_frames",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _value(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if isinstance(value, dict):
        return value.get("total")
    return value


def build_comparison_rows(
    root: str | Path,
    route_a_run: str = "scene1_opera_a",
    route_b_run: str = "scene1_opera_b_joint_av",
    route_b_no_warmup_run: str = "scene1_opera_b_joint_av_no_warmup",
) -> list[dict[str, Any]]:
    root = Path(root)
    route_a_audio = _load_json(root / f"runs/{route_a_run}/eval/audio_summary.json")
    route_a_visual = _load_json(root / f"runs/{route_a_run}/ftgspp/summary.json")
    route_b_audio = _load_json(root / f"runs/{route_b_run}/eval/audio_summary.json")
    route_b_visual = _load_json(root / f"runs/{route_b_run}/eval/visual_summary.json")
    route_b_nowarm_audio = _load_json(
        root / f"runs/{route_b_no_warmup_run}/eval/audio_summary.json"
    )
    route_b_nowarm_visual = _load_json(
        root / f"runs/{route_b_no_warmup_run}/eval/visual_summary.json"
    )
    return [
        {
            "method": "Route A frozen carrier",
            "schedule": "FTGS++ visual pretrain -> frozen carrier AudioGS-style audio",
            "audio_protocol": "AudioGS 3s heldout crop",
            "visual_protocol": "FTGS++ summary",
            "MAG": route_a_audio.get("MAG"),
            "ENV": route_a_audio.get("ENV"),
            "LRE": route_a_audio.get("LRE"),
            "DPAM": route_a_audio.get("DPAM"),
            "DPAM_available": route_a_audio.get("DPAM_available"),
            "DPAM_error": route_a_audio.get("DPAM_error"),
            "RTE": route_a_audio.get("RTE"),
            "RTE_available": route_a_audio.get("RTE_available"),
            "PSNR": _value(route_a_visual.get("PSNR_mean", {}), "total"),
            "SSIM-1": _value(route_a_visual.get("SSIM-1", {}), "total"),
            "LPIPS-Alex": _value(route_a_visual.get("LPIPS-Alex", {}), "total"),
            "audio_camera": route_a_audio.get("camera"),
            "visual_camera": route_a_audio.get("camera"),
            "audio_windows": 1,
            "audio_window_seconds": 3.0,
            "visual_frames": None,
        },
        {
            "method": "Route B joint AV",
            "schedule": "visual init -> audio warmup -> strict AV joint finetune",
            "audio_protocol": "strict AV aligned 0.5s window average",
            "visual_protocol": "joint visual eval",
            "MAG": route_b_audio.get("MAG"),
            "ENV": route_b_audio.get("ENV"),
            "LRE": route_b_audio.get("LRE"),
            "DPAM": route_b_audio.get("DPAM"),
            "DPAM_available": route_b_audio.get("DPAM_available"),
            "DPAM_error": route_b_audio.get("DPAM_error"),
            "RTE": route_b_audio.get("RTE"),
            "RTE_available": route_b_audio.get("RTE_available"),
            "PSNR": route_b_visual.get("PSNR"),
            "SSIM-1": None,
            "LPIPS-Alex": None,
            "audio_camera": route_b_audio.get("camera"),
            "visual_camera": route_b_visual.get("camera"),
            "audio_windows": route_b_audio.get("num_windows"),
            "audio_window_seconds": route_b_audio.get("audio_window_seconds"),
            "visual_frames": route_b_visual.get("num_frames"),
        },
        {
            "method": "Route B joint AV no warmup",
            "schedule": "visual init -> strict AV joint finetune from step 0",
            "audio_protocol": "strict AV aligned 0.5s window average",
            "visual_protocol": "joint visual eval",
            "MAG": route_b_nowarm_audio.get("MAG"),
            "ENV": route_b_nowarm_audio.get("ENV"),
            "LRE": route_b_nowarm_audio.get("LRE"),
            "DPAM": route_b_nowarm_audio.get("DPAM"),
            "DPAM_available": route_b_nowarm_audio.get("DPAM_available"),
            "DPAM_error": route_b_nowarm_audio.get("DPAM_error"),
            "RTE": route_b_nowarm_audio.get("RTE"),
            "RTE_available": route_b_nowarm_audio.get("RTE_available"),
            "PSNR": route_b_nowarm_visual.get("PSNR"),
            "SSIM-1": None,
            "LPIPS-Alex": None,
            "audio_camera": route_b_nowarm_audio.get("camera"),
            "visual_camera": route_b_nowarm_visual.get("camera"),
            "audio_windows": route_b_nowarm_audio.get("num_windows"),
            "audio_window_seconds": route_b_nowarm_audio.get("audio_window_seconds"),
            "visual_frames": route_b_nowarm_visual.get("num_frames"),
        },
    ]


def _format_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_comparison_tables(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in METRIC_COLUMNS})
    lines = [
        "| " + " | ".join(METRIC_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(METRIC_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(key)) for key in METRIC_COLUMNS) + " |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")


def _load_route_a_params(checkpoint: dict) -> AcousticGaussianParameters:
    state = checkpoint["params"]
    params = AcousticGaussianParameters(num_points=int(state["mono_gain"].shape[0]))
    with torch.no_grad():
        params.mono_gain.copy_(state["mono_gain"])
        params.diff_gain.copy_(state["diff_gain"])
    return params


def export_route_a_audio(manifest_path: Path, checkpoint_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = AudioCropDataset(manifest_path, split="eval")
    if len(dataset) != 1:
        raise ValueError(f"expected one Route A eval camera, got {len(dataset)}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    carrier = FrozenVisualCarrier.load(checkpoint["carrier_path"])
    acoustic_carrier = select_topk_acoustic_carrier(carrier, t=0.0, top_k=int(checkpoint["top_k"]))
    params = _load_route_a_params(checkpoint)
    sample = dataset[0]
    with torch.no_grad():
        pred = render_audio(acoustic_carrier, params, sample["source_audio"]).detach().cpu()
    sample_rate = int(dataset.manifest.audio.sample_rate)
    sf.write(output_dir / "reconstruction.wav", pred.T.numpy(), sample_rate)
    sf.write(output_dir / "target.wav", sample["target_audio"].T.numpy(), sample_rate)
    sf.write(output_dir / "source.wav", sample["source_audio"].T.numpy(), sample_rate)


def _frame_hop_samples(manifest: SceneManifest, sample_rate: int) -> int:
    if len(manifest.frame_times) >= 2:
        deltas = np.diff(np.asarray(manifest.frame_times, dtype=np.float64))
        seconds = float(np.median(deltas))
    else:
        seconds = 1.0 / float(manifest.fps)
    return max(1, int(round(seconds * sample_rate)))


def _center_segment(audio: torch.Tensor, segment_samples: int) -> torch.Tensor:
    center = int(audio.shape[-1]) // 2
    start = max(0, center - segment_samples // 2)
    end = min(int(audio.shape[-1]), start + segment_samples)
    if end - start < segment_samples:
        start = max(0, end - segment_samples)
    return audio[:, start:end]


def export_joint_audio_sequence(
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    max_frames: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or {}
    window_seconds = float(config.get("audio_window_seconds", 0.5))
    visual_scale = float(config.get("visual_scale", 0.125))
    memmap_root = config.get("ftgspp_memmap")
    dataset = VisualFrameDataset(
        manifest_path,
        split="eval",
        scale=visual_scale,
        memmap_root=memmap_root,
    )
    audio_cropper = TimedAudioCropper(manifest_path, crop_seconds=window_seconds, mode="center")
    sample_rate = int(audio_cropper.manifest.audio.sample_rate)
    segment_samples = _frame_hop_samples(audio_cropper.manifest, sample_rate)
    limit = len(dataset) if max_frames is None else min(int(max_frames), len(dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _restore_joint_model(checkpoint).to(device)
    pred_segments = []
    target_segments = []
    source_segments = []
    with torch.no_grad():
        for idx in range(limit):
            visual_sample = dataset[idx]
            audio_sample = audio_cropper.get_crop(str(visual_sample["camera"]), visual_sample["time"])
            pred = model.render_audio(
                visual_sample["time"].to(device),
                audio_sample["source_audio"].to(device),
            ).detach().cpu()
            pred_segments.append(_center_segment(pred, segment_samples))
            target_segments.append(_center_segment(audio_sample["target_audio"], segment_samples))
            source_segments.append(_center_segment(audio_sample["source_audio"], segment_samples))
    sf.write(output_dir / "reconstruction.wav", torch.cat(pred_segments, dim=-1).T.numpy(), sample_rate)
    sf.write(output_dir / "target.wav", torch.cat(target_segments, dim=-1).T.numpy(), sample_rate)
    sf.write(output_dir / "source.wav", torch.cat(source_segments, dim=-1).T.numpy(), sample_rate)


def _write_ppm(path: Path, image: torch.Tensor) -> None:
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    array = (image.numpy() * 255.0).round().astype(np.uint8)
    header = f"P6\n{array.shape[1]} {array.shape[0]}\n255\n".encode("ascii")
    path.write_bytes(header + array.tobytes())


def _encode_mp4(frame_dir: Path, output_path: Path, fps: float) -> None:
    if shutil.which("ffmpeg") is None:
        return
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps}",
            "-i",
            str(frame_dir / "frame_%05d.ppm"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )


def export_visual_sequence(
    manifest_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    route: str,
    scale: float = 0.125,
    max_frames: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = output_dir / "target_frames"
    pred_dir = output_dir / "reconstruction_frames"
    target_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    manifest = SceneManifest.load(manifest_path)
    visual_root = Path(manifest.visual_root)
    memmap_root = visual_root if (visual_root / "meta.json").exists() else None
    dataset = VisualFrameDataset(
        manifest_path,
        split="eval",
        scale=scale,
        memmap_root=memmap_root,
    )
    limit = len(dataset) if max_frames is None else min(int(max_frames), len(dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if route == "route_a":
        renderer = FTGSRendererBridge.load_checkpoint(checkpoint_path).to(device)
        render = lambda batch: renderer.render_rgb(batch).clamp(0.0, 1.0)
    elif route == "route_b":
        model = _restore_joint_model(torch.load(checkpoint_path, map_location="cpu", weights_only=False)).to(device)
        render = lambda batch: model.render_rgb(batch).clamp(0.0, 1.0)
    else:
        raise ValueError(f"unknown visual route {route!r}")

    with torch.no_grad():
        for idx in range(limit):
            sample = dataset[idx]
            batch = _move_tensor_values(sample, device)
            pred = render(batch)
            _write_ppm(pred_dir / f"frame_{idx:05d}.ppm", pred)
            _write_ppm(target_dir / f"frame_{idx:05d}.ppm", batch["target_rgb"])
    fps = float(manifest.fps)
    _encode_mp4(pred_dir, output_dir / "reconstruction.mp4", fps)
    _encode_mp4(target_dir, output_dir / "target.mp4", fps)


def export_artifacts(root: str | Path, output_dir: str | Path, max_video_frames: int | None) -> None:
    export_artifacts_for_runs(
        root=root,
        output_dir=output_dir,
        route_a_run="scene1_opera_a",
        route_b_run="scene1_opera_b_joint_av",
        route_b_no_warmup_run="scene1_opera_b_joint_av_no_warmup",
        ftgspp_scene="scene1_opera",
        max_video_frames=max_video_frames,
    )


def export_artifacts_for_runs(
    root: str | Path,
    output_dir: str | Path,
    route_a_run: str,
    route_b_run: str,
    route_b_no_warmup_run: str,
    ftgspp_scene: str,
    max_video_frames: int | None,
) -> None:
    root = Path(root)
    output_dir = Path(output_dir)
    manifest = root / f"runs/{route_a_run}/scene_manifest.json"
    specs = [
        (
            "route_a_frozen_carrier",
            root / f"runs/{route_a_run}/stage2_audio.pt",
            root / f"runs/{route_a_run}/ftgspp/{ftgspp_scene}/00/gaussians.pt",
            "route_a",
        ),
        (
            "route_b_warmup_joint",
            root / f"runs/{route_b_run}/joint_finetune.pt",
            root / f"runs/{route_b_run}/joint_finetune.pt",
            "route_b",
        ),
        (
            "route_b_no_warmup_joint",
            root / f"runs/{route_b_no_warmup_run}/joint_finetune.pt",
            root / f"runs/{route_b_no_warmup_run}/joint_finetune.pt",
            "route_b",
        ),
    ]
    export_route_a_audio(manifest, specs[0][1], output_dir / specs[0][0] / "audio")
    for name, audio_checkpoint, visual_checkpoint, route in specs[1:]:
        export_joint_audio_sequence(
            manifest,
            audio_checkpoint,
            output_dir / name / "audio",
            max_frames=max_video_frames,
        )
    for name, _, visual_checkpoint, route in specs:
        export_visual_sequence(
            manifest,
            output_dir / name / "video",
            visual_checkpoint,
            route=route,
            max_frames=max_video_frames,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect scene1_opera Route A/B comparison artifacts.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="AVGaussianFusion repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <root>/runs/scene1_opera_comparison.",
    )
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--max-video-frames", type=int)
    parser.add_argument("--route-a-run", default="scene1_opera_a")
    parser.add_argument("--route-b-run", default="scene1_opera_b_joint_av")
    parser.add_argument("--route-b-no-warmup-run", default="scene1_opera_b_joint_av_no_warmup")
    parser.add_argument("--ftgspp-scene", default="scene1_opera")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "runs/scene1_opera_comparison"
    rows = build_comparison_rows(
        root,
        route_a_run=args.route_a_run,
        route_b_run=args.route_b_run,
        route_b_no_warmup_run=args.route_b_no_warmup_run,
    )
    write_comparison_tables(rows, output_dir / "metrics")
    if not args.skip_artifacts:
        export_artifacts_for_runs(
            root=root,
            output_dir=output_dir,
            route_a_run=args.route_a_run,
            route_b_run=args.route_b_run,
            route_b_no_warmup_run=args.route_b_no_warmup_run,
            ftgspp_scene=args.ftgspp_scene,
            max_video_frames=args.max_video_frames,
        )
    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
