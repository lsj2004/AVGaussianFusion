from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
import soundfile as sf

from avfusion.audio import stft_magnitude_loss
from avfusion.data.audio_video_dataset import AudioCropDataset, TimedAudioCropper
from avfusion.data.manifest import SceneManifest
from avfusion.data.visual_frame_dataset import FrameReader, VisualFrameDataset
from avfusion.joint.audio_head import JointAudioHead, SpectralJointAudioHead
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.joint.losses import geometry_regularization
from avfusion.joint.model import JointAVGaussianModel


@dataclass(frozen=True)
class TrainingConfig:
    manifest: str
    ftgspp_checkpoint: str
    ftgspp_memmap: str | None
    output: str
    warmup_steps: int
    joint_steps: int
    top_k: int
    audio_lr: float
    shared_lr: float
    geometry_reg_weight: float
    rgb_loss_weight: float
    audio_loss_weight: float
    visual_scale: float
    audio_window_seconds: float
    audio_head_type: str
    config: str | None = None


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be nonnegative, got {parsed}")
    return parsed


def _top_k(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError(f"must be at least 2, got {parsed}")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize Route B joint AV Gaussian fine-tuning from an FTGS++ checkpoint."
    )
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument("--ftgspp-checkpoint")
    parser.add_argument("--ftgspp-memmap")
    parser.add_argument("--output")
    parser.add_argument("--warmup-steps", type=_nonnegative_int)
    parser.add_argument("--joint-steps", type=_nonnegative_int)
    parser.add_argument("--top-k", type=_top_k)
    parser.add_argument("--audio-lr", type=float)
    parser.add_argument("--shared-lr", type=float)
    parser.add_argument("--audio-window-seconds", type=float)
    parser.add_argument("--audio-head-type", choices=["simple", "spectral"])
    return parser


def _load_yaml_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    loaded = yaml.safe_load(Path(config_path).read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Route B config must be a mapping, got {type(loaded).__name__}")
    return loaded


def _config_path_value(config: dict[str, Any], key: str) -> str | None:
    paths = config.get("paths") or {}
    if not isinstance(paths, dict):
        raise ValueError("Route B config field paths must be a mapping")
    value = paths.get(key)
    if value is None:
        return None
    return str(value)


def _config_train_value(config: dict[str, Any], key: str, default: Any) -> Any:
    train = config.get("train") or {}
    if not isinstance(train, dict):
        raise ValueError("Route B config field train must be a mapping")
    return train.get(key, default)


def _config_loss_value(config: dict[str, Any], key: str, default: Any) -> Any:
    losses = config.get("losses") or {}
    if not isinstance(losses, dict):
        raise ValueError("Route B config field losses must be a mapping")
    return losses.get(key, default)


def _require_value(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"missing required Route B setting: {name}")
    return value


def resolve_training_config(args: argparse.Namespace) -> TrainingConfig:
    config = _load_yaml_config(args.config)

    manifest = args.manifest or _config_path_value(config, "manifest")
    ftgspp_checkpoint = args.ftgspp_checkpoint or _config_path_value(config, "ftgspp_checkpoint")
    ftgspp_memmap = args.ftgspp_memmap or _config_path_value(config, "ftgspp_memmap")
    output = args.output or _config_path_value(config, "output_checkpoint")
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = _config_train_value(config, "warmup_steps", 1000)
    joint_steps = args.joint_steps
    if joint_steps is None:
        joint_steps = _config_train_value(config, "joint_steps", 1000)
    top_k = args.top_k
    if top_k is None:
        top_k = _config_train_value(config, "top_k", 8192)
    audio_lr = args.audio_lr
    if audio_lr is None:
        audio_lr = _config_train_value(config, "audio_lr", 5e-4)
    shared_lr = args.shared_lr
    if shared_lr is None:
        shared_lr = _config_train_value(config, "shared_lr", 1e-5)
    geometry_reg_weight = _config_loss_value(config, "geometry_reg_weight", 1e-3)
    rgb_loss_weight = _config_loss_value(config, "visual_weight", 1.0)
    audio_loss_weight = _config_loss_value(config, "audio_weight", 1.0)
    visual_scale = _config_train_value(config, "visual_scale", 0.125)
    audio_window_seconds = args.audio_window_seconds
    if audio_window_seconds is None:
        audio_window_seconds = _config_train_value(config, "audio_window_seconds", 0.5)
    audio_head_type = args.audio_head_type
    if audio_head_type is None:
        audio_head_type = _config_train_value(config, "audio_head_type", "simple")

    warmup_steps = int(_require_value(warmup_steps, "train.warmup_steps"))
    joint_steps = int(_require_value(joint_steps, "train.joint_steps"))
    top_k = int(_require_value(top_k, "train.top_k"))
    audio_lr = float(_require_value(audio_lr, "train.audio_lr"))
    shared_lr = float(_require_value(shared_lr, "train.shared_lr"))
    geometry_reg_weight = float(
        _require_value(geometry_reg_weight, "losses.geometry_reg_weight")
    )
    rgb_loss_weight = float(_require_value(rgb_loss_weight, "losses.visual_weight"))
    audio_loss_weight = float(_require_value(audio_loss_weight, "losses.audio_weight"))
    visual_scale = float(_require_value(visual_scale, "train.visual_scale"))
    audio_window_seconds = float(
        _require_value(audio_window_seconds, "train.audio_window_seconds")
    )
    audio_head_type = str(_require_value(audio_head_type, "train.audio_head_type"))
    if warmup_steps < 0:
        raise ValueError(f"train.warmup_steps must be nonnegative, got {warmup_steps}")
    if joint_steps < 0:
        raise ValueError(f"train.joint_steps must be nonnegative, got {joint_steps}")
    if top_k < 2:
        raise ValueError(f"train.top_k must be at least 2, got {top_k}")
    if geometry_reg_weight < 0:
        raise ValueError(
            f"losses.geometry_reg_weight must be nonnegative, got {geometry_reg_weight}"
        )
    if rgb_loss_weight < 0:
        raise ValueError(f"losses.visual_weight must be nonnegative, got {rgb_loss_weight}")
    if joint_steps > 0 and rgb_loss_weight <= 0:
        raise ValueError("losses.visual_weight must be positive for joint AV training")
    if audio_loss_weight < 0:
        raise ValueError(f"losses.audio_weight must be nonnegative, got {audio_loss_weight}")
    if visual_scale <= 0:
        raise ValueError(f"train.visual_scale must be positive, got {visual_scale}")
    if audio_window_seconds <= 0:
        raise ValueError(
            f"train.audio_window_seconds must be positive, got {audio_window_seconds}"
        )
    if audio_head_type not in {"simple", "spectral"}:
        raise ValueError(
            f"train.audio_head_type must be one of simple/spectral, got {audio_head_type!r}"
        )

    return TrainingConfig(
        manifest=str(_require_value(manifest, "paths.manifest")),
        ftgspp_checkpoint=str(_require_value(ftgspp_checkpoint, "paths.ftgspp_checkpoint")),
        ftgspp_memmap=str(ftgspp_memmap) if ftgspp_memmap is not None else None,
        output=str(_require_value(output, "paths.output_checkpoint")),
        warmup_steps=warmup_steps,
        joint_steps=joint_steps,
        top_k=top_k,
        audio_lr=audio_lr,
        shared_lr=shared_lr,
        geometry_reg_weight=geometry_reg_weight,
        rgb_loss_weight=rgb_loss_weight,
        audio_loss_weight=audio_loss_weight,
        visual_scale=visual_scale,
        audio_window_seconds=audio_window_seconds,
        audio_head_type=audio_head_type,
        config=args.config,
    )


def build_model(
    ftgspp_checkpoint: str | Path,
    top_k: int = 8192,
    audio_head_type: str = "simple",
) -> JointAVGaussianModel:
    bridge = FTGSRendererBridge.load_checkpoint(ftgspp_checkpoint)
    means = bridge.gaussians.means
    num_points = int(means.shape[0])
    if num_points < 2:
        raise ValueError(f"Route B joint training requires at least two gaussians, got {num_points}")
    effective_top_k = min(int(top_k), num_points)
    if audio_head_type == "simple":
        audio_head = JointAudioHead(num_points=num_points, top_k=effective_top_k)
    elif audio_head_type == "spectral":
        audio_head = SpectralJointAudioHead(num_points=num_points, top_k=effective_top_k)
    else:
        raise ValueError(f"unknown audio_head_type {audio_head_type!r}")
    return JointAVGaussianModel(bridge, audio_head)


def _move_tensor_values(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def save_joint_checkpoint(
    output_path: str | Path,
    model: JointAVGaussianModel,
    stage: str,
    ftgspp_checkpoint: str | Path,
    manifest_path: str | Path,
    config: dict[str, Any],
    loss_history: list[float | dict[str, float]],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": "B_joint_av",
            "stage": stage,
            "ftgspp_checkpoint": str(ftgspp_checkpoint),
            "manifest_path": str(manifest_path),
            "shared_gaussians": model.shared_gaussians.state_dict(),
            "audio_head": model.audio_head.state_dict(),
            "config": dict(config),
            "loss_history": list(loss_history),
        },
        output_path,
    )


def _audio_frames(path: str | Path) -> int:
    return int(sf.info(str(path)).frames)


def _compute_valid_aligned_samples(
    manifest_path: str | Path,
    audio_window_seconds: float,
) -> dict[str, int | None]:
    manifest = SceneManifest.load(manifest_path)
    sample_rate = int(manifest.audio.sample_rate)
    crop_samples = int(round(float(audio_window_seconds) * sample_rate))
    source_frames = _audio_frames(manifest.audio.source_path)
    per_camera_valid = []
    for camera in manifest.train_cameras:
        target_frames = _audio_frames(manifest.cameras[camera].audio_path)
        max_frames = min(source_frames, target_frames)
        valid = 0
        for time_seconds in manifest.frame_times:
            center_sample = int(round(float(time_seconds) * sample_rate))
            start_sample = center_sample - crop_samples // 2
            end_sample = start_sample + crop_samples
            if start_sample >= 0 and end_sample <= max_frames:
                valid += 1
        per_camera_valid.append(valid)
    unique_valid_counts = sorted(set(per_camera_valid))
    return {
        "valid_aligned_samples": int(sum(per_camera_valid)),
        "valid_frames_per_camera": (
            int(unique_valid_counts[0]) if len(unique_valid_counts) == 1 else None
        ),
    }


def write_train_summary(
    output_path: str | Path,
    manifest_path: str | Path,
    summary: dict[str, float | int | str],
    audio_window_seconds: float,
) -> None:
    manifest = SceneManifest.load(manifest_path)
    valid = _compute_valid_aligned_samples(
        manifest_path,
        audio_window_seconds=audio_window_seconds,
    )
    payload = {
        **summary,
        "manifest": str(manifest_path),
        "train_cams": len(manifest.train_cameras),
        "eval_cam": manifest.eval_cameras[0] if manifest.eval_cameras else None,
        "source_audio": Path(manifest.audio.source_path).name,
        "audio_window_seconds": float(audio_window_seconds),
        "valid_aligned_samples": int(valid["valid_aligned_samples"]),
        "valid_frames_per_camera": valid["valid_frames_per_camera"],
    }
    summary_path = Path(output_path).parent / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train_audio_warmup(
    model: JointAVGaussianModel,
    manifest_path: str | Path,
    steps: int,
    lr: float,
) -> list[float]:
    if steps <= 0:
        return []
    dataset = AudioCropDataset(manifest_path, split="train")
    if len(dataset) == 0:
        raise ValueError("training split is empty")

    model.freeze_shared()
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.audio_head.parameters(), lr=lr)
    loss_history: list[float] = []
    render_time = torch.tensor([[0.0]], device=device)

    for step in range(steps):
        sample = dataset[step % len(dataset)]
        optimizer.zero_grad(set_to_none=True)
        pred = model.render_audio(render_time, sample["source_audio"].to(device))
        target = sample["target_audio"].to(pred)
        loss = stft_magnitude_loss(pred, target)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))

    return loss_history


def train_joint_finetune(
    model: JointAVGaussianModel,
    manifest_path: str | Path,
    steps: int,
    shared_lr: float,
    audio_lr: float,
    geometry_reg_weight: float,
    rgb_loss_weight: float,
    audio_loss_weight: float,
    visual_scale: float,
    audio_window_seconds: float = 0.5,
    ftgspp_memmap: str | Path | None = None,
    frame_reader: FrameReader | None = None,
) -> list[dict[str, float]]:
    if steps <= 0:
        return []
    if rgb_loss_weight <= 0:
        raise ValueError("rgb_loss_weight must be positive for joint AV fine-tuning")
    dataset = AudioCropDataset(manifest_path, split="train")
    if len(dataset) == 0:
        raise ValueError("training split is empty")
    visual_dataset = None
    if rgb_loss_weight > 0:
        visual_dataset = VisualFrameDataset(
            manifest_path,
            split="train",
            scale=visual_scale,
            memmap_root=ftgspp_memmap,
            frame_reader=frame_reader,
        )
        if len(visual_dataset) == 0:
            raise ValueError("visual training split is empty")
    timed_audio_cropper = TimedAudioCropper(
        manifest_path,
        crop_seconds=audio_window_seconds,
        mode="center",
        allow_padding=False,
    )

    model.unfreeze_shared_geometry()
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(
        model.parameter_groups(shared_lr=shared_lr, audio_lr=audio_lr)
    )
    loss_history: list[dict[str, float]] = []
    visual_cursor = 0

    for step in range(steps):
        visual_sample = None
        if visual_dataset is not None:
            audio_sample = None
            for _ in range(len(visual_dataset)):
                candidate = visual_dataset[visual_cursor % len(visual_dataset)]
                visual_cursor += 1
                try:
                    audio_sample = timed_audio_cropper.get_crop(
                        camera=str(candidate["camera"]),
                        time_seconds=candidate["time"],
                    )
                    visual_sample = candidate
                    break
                except ValueError as error:
                    if "padding" not in str(error):
                        raise
            if visual_sample is None or audio_sample is None:
                raise ValueError("no non-padding visual/audio-aligned training samples available")
            render_time = visual_sample["time"].to(device)
        else:
            sample = dataset[step % len(dataset)]
            audio_sample = {
                "source_audio": sample["source_audio"],
                "target_audio": sample["target_audio"],
            }
            render_time = torch.tensor([[0.0]], device=device)
        optimizer.zero_grad(set_to_none=True)
        pred = model.render_audio(render_time, audio_sample["source_audio"].to(device))
        target = audio_sample["target_audio"].to(pred)
        audio_loss = stft_magnitude_loss(pred, target)
        rgb_loss = audio_loss.new_zeros(())
        if visual_sample is not None:
            visual_sample = _move_tensor_values(
                visual_sample,
                device,
            )
            pred_rgb = model.render_rgb(visual_sample)
            rgb_loss = torch.nn.functional.l1_loss(
                pred_rgb,
                visual_sample["target_rgb"].to(pred_rgb),
            )
        geo_loss = geometry_regularization(model)
        total = (
            float(rgb_loss_weight) * rgb_loss
            + float(audio_loss_weight) * audio_loss
            + float(geometry_reg_weight) * geo_loss
        )
        total.backward()
        optimizer.step()
        loss_history.append(
            {
                "total": float(total.detach().cpu().item()),
                "rgb": float(rgb_loss.detach().cpu().item()),
                "audio": float(audio_loss.detach().cpu().item()),
                "geo": float(geo_loss.detach().cpu().item()),
                "camera": str(visual_sample["camera"]) if visual_sample is not None else str(audio_sample.get("camera", "")),
                "frame": int(visual_sample["frame"]) if visual_sample is not None else -1,
                "time": float(render_time.detach().cpu().reshape(-1)[0].item()),
                "start_sample": int(audio_sample.get("start_sample", -1)),
            }
        )

    return loss_history


def train_and_save(
    manifest_path: str | Path,
    ftgspp_checkpoint: str | Path,
    output_path: str | Path,
    warmup_steps: int,
    joint_steps: int,
    top_k: int,
    audio_lr: float,
    shared_lr: float,
    geometry_reg_weight: float = 1e-3,
    rgb_loss_weight: float = 1.0,
    audio_loss_weight: float = 1.0,
    visual_scale: float = 0.125,
    audio_window_seconds: float = 0.5,
    audio_head_type: str = "simple",
    ftgspp_memmap: str | Path | None = None,
    config_path: str | Path | None = None,
    frame_reader: FrameReader | None = None,
) -> dict[str, float | int | str]:
    model = build_model(ftgspp_checkpoint, top_k=top_k, audio_head_type=audio_head_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    warmup_loss_history = train_audio_warmup(
        model=model,
        manifest_path=manifest_path,
        steps=warmup_steps,
        lr=audio_lr,
    )
    joint_loss_history = train_joint_finetune(
        model=model,
        manifest_path=manifest_path,
        steps=joint_steps,
        shared_lr=shared_lr,
        audio_lr=audio_lr,
        geometry_reg_weight=geometry_reg_weight,
        rgb_loss_weight=rgb_loss_weight,
        audio_loss_weight=audio_loss_weight,
        visual_scale=visual_scale,
        audio_window_seconds=audio_window_seconds,
        ftgspp_memmap=ftgspp_memmap,
        frame_reader=frame_reader,
    )
    implemented_stages = []
    if warmup_steps > 0:
        implemented_stages.append("audio_warmup")
    if joint_steps > 0:
        implemented_stages.append("joint_finetune")
    stage = implemented_stages[-1] if implemented_stages else "initialized"
    loss_history: list[float | dict[str, float]] = [
        *warmup_loss_history,
        *joint_loss_history,
    ]
    config = {
        "manifest": str(manifest_path),
        "ftgspp_checkpoint": str(ftgspp_checkpoint),
        "ftgspp_memmap": str(ftgspp_memmap) if ftgspp_memmap is not None else None,
        "output": str(output_path),
        "warmup_steps": int(warmup_steps),
        "joint_steps": int(joint_steps),
        "top_k": int(top_k),
        "audio_lr": float(audio_lr),
        "shared_lr": float(shared_lr),
        "geometry_reg_weight": float(geometry_reg_weight),
        "rgb_loss_weight": float(rgb_loss_weight),
        "audio_loss_weight": float(audio_loss_weight),
        "visual_scale": float(visual_scale),
        "audio_window_seconds": float(audio_window_seconds),
        "audio_head_type": str(audio_head_type),
        "config": str(config_path) if config_path is not None else None,
        "implemented_stages": implemented_stages,
        "warmup_loss_history": warmup_loss_history,
        "joint_loss_history": joint_loss_history,
    }
    save_joint_checkpoint(
        output_path=output_path,
        model=model,
        stage=stage,
        ftgspp_checkpoint=ftgspp_checkpoint,
        manifest_path=manifest_path,
        config=config,
        loss_history=loss_history,
    )
    summary = {
        "route": "B_joint_av",
        "stage": stage,
        "steps": int(warmup_steps + joint_steps),
        "warmup_steps": int(warmup_steps),
        "joint_steps": int(joint_steps),
        "audio_head_type": str(audio_head_type),
        "final_loss": (
            joint_loss_history[-1]["total"]
            if joint_loss_history
            else warmup_loss_history[-1]
            if warmup_loss_history
            else 0.0
        ),
        "output": str(output_path),
    }
    write_train_summary(
        output_path=output_path,
        manifest_path=manifest_path,
        summary=summary,
        audio_window_seconds=audio_window_seconds,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = resolve_training_config(args)
    summary = train_and_save(
        manifest_path=config.manifest,
        ftgspp_checkpoint=config.ftgspp_checkpoint,
        output_path=config.output,
        warmup_steps=config.warmup_steps,
        joint_steps=config.joint_steps,
        top_k=config.top_k,
        audio_lr=config.audio_lr,
        shared_lr=config.shared_lr,
        geometry_reg_weight=config.geometry_reg_weight,
        rgb_loss_weight=config.rgb_loss_weight,
        audio_loss_weight=config.audio_loss_weight,
        visual_scale=config.visual_scale,
        audio_window_seconds=config.audio_window_seconds,
        audio_head_type=config.audio_head_type,
        ftgspp_memmap=config.ftgspp_memmap,
        config_path=config.config,
    )
    print({**summary, "config": asdict(config)})


if __name__ == "__main__":
    main()
