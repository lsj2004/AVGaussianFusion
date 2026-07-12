from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from avfusion.audio import audiogs_mono_diff_loss, stft_magnitude_loss
from avfusion.data.audio_video_dataset import TimedAudioCropper
from avfusion.data.visual_frame_dataset import FrameReader, VisualFrameDataset
from avfusion.joint.ftgspp_bridge import FTGSRendererBridge
from avfusion.soft.acoustic_field import AcousticGaussianField
from avfusion.soft.frequency_transfer_renderer import FrequencyTransferRenderer
from avfusion.soft.losses import soft_coupling_losses


@dataclass(frozen=True)
class SoftTrainingConfig:
    manifest: str
    ftgspp_checkpoint: str
    output: str
    acoustic_steps: int
    joint_steps: int
    num_acoustic_points: int
    top_k: int
    audio_lr: float
    acoustic_lr: float
    shared_lr: float
    audio_window_seconds: float
    audio_crop_mode: str
    anchored_fraction: float
    dynamic_fraction: float
    anchor_weight: float
    motion_weight: float
    activity_weight: float
    sparse_weight: float
    rgb_weight: float
    audio_weight: float
    visual_guard_psnr_drop_db: float
    audio_guard_relative_drop: float
    ftgspp_memmap: str | None = None
    visual_scale: float = 0.125
    audio_loss_type: str = "audiogs_mono_diff"
    audio_diff_weight: float = 0.5
    audio_lre_loss_weight: float = 0.075
    audio_bandpass: dict[str, float | bool] | None = None
    config: str | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Route C soft-coupled AV Gaussian fields.")
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument("--ftgspp-checkpoint")
    parser.add_argument("--ftgspp-memmap")
    parser.add_argument("--output")
    parser.add_argument("--acoustic-steps", type=int)
    parser.add_argument("--joint-steps", type=int)
    parser.add_argument("--num-acoustic-points", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--audio-window-seconds", type=float)
    return parser


def _load_yaml_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    loaded = yaml.safe_load(Path(config_path).read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Route C config must be a mapping, got {type(loaded).__name__}")
    return loaded


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Route C config field {name} must be a mapping")
    return value


def _path_value(config: dict[str, Any], key: str) -> str | None:
    value = _section(config, "paths").get(key)
    return None if value is None else str(value)


def _value(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return _section(config, section).get(key, default)


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise ValueError(f"missing required Route C config value {name}")
    return value


def resolve_soft_training_config(args: argparse.Namespace) -> SoftTrainingConfig:
    config = _load_yaml_config(args.config)
    train = _section(config, "train")
    losses = _section(config, "losses")
    guards = _section(config, "guards")
    audio_bandpass = losses.get("audio_bandpass")

    audio_window_seconds = float(
        args.audio_window_seconds
        if args.audio_window_seconds is not None
        else train.get("audio_window_seconds", 0.5)
    )
    if audio_window_seconds > 1.0 and not bool(train.get("allow_long_audio_window", False)):
        raise ValueError(
            "audio_window_seconds must be <= 1.0 for Route C quasi-static alignment; "
            "set train.allow_long_audio_window=true only for explicit ablations"
        )

    cfg = SoftTrainingConfig(
        manifest=_required(args.manifest or _path_value(config, "manifest"), "paths.manifest"),
        ftgspp_checkpoint=_required(
            args.ftgspp_checkpoint or _path_value(config, "ftgspp_checkpoint"),
            "paths.ftgspp_checkpoint",
        ),
        output=_required(args.output or _path_value(config, "output_checkpoint"), "paths.output_checkpoint"),
        ftgspp_memmap=args.ftgspp_memmap or _path_value(config, "ftgspp_memmap"),
        acoustic_steps=int(
            args.acoustic_steps if args.acoustic_steps is not None else train.get("acoustic_steps", 3000)
        ),
        joint_steps=int(args.joint_steps if args.joint_steps is not None else train.get("joint_steps", 1000)),
        num_acoustic_points=int(
            args.num_acoustic_points
            if args.num_acoustic_points is not None
            else train.get("num_acoustic_points", 8192)
        ),
        top_k=int(args.top_k if args.top_k is not None else train.get("top_k", 8192)),
        audio_lr=float(train.get("audio_lr", 5e-4)),
        acoustic_lr=float(train.get("acoustic_lr", train.get("audio_lr", 5e-4))),
        shared_lr=float(train.get("shared_lr", 0.0)),
        audio_window_seconds=audio_window_seconds,
        audio_crop_mode=str(train.get("audio_crop_mode", "center")),
        anchored_fraction=float(train.get("anchored_fraction", 0.6)),
        dynamic_fraction=float(train.get("dynamic_fraction", 0.2)),
        anchor_weight=float(losses.get("anchor_weight", 1e-3)),
        motion_weight=float(losses.get("motion_weight", 1e-3)),
        activity_weight=float(losses.get("activity_weight", 1e-4)),
        sparse_weight=float(losses.get("sparse_weight", 1e-5)),
        rgb_weight=float(losses.get("visual_weight", 1.0)),
        audio_weight=float(losses.get("audio_weight", 1.0)),
        visual_guard_psnr_drop_db=float(guards.get("visual_psnr_drop_db", 0.3)),
        audio_guard_relative_drop=float(guards.get("audio_relative_drop", 0.05)),
        visual_scale=float(train.get("visual_scale", 0.125)),
        audio_loss_type=str(train.get("audio_loss_type", "audiogs_mono_diff")),
        audio_diff_weight=float(losses.get("audio_diff_weight", 0.5)),
        audio_lre_loss_weight=float(losses.get("audio_lre_loss_weight", 0.075)),
        audio_bandpass=dict(audio_bandpass) if isinstance(audio_bandpass, dict) else None,
        config=args.config,
    )
    if cfg.audio_crop_mode not in {"center", "start"}:
        raise ValueError(f"audio_crop_mode must be center or start, got {cfg.audio_crop_mode!r}")
    if cfg.num_acoustic_points < 2 or cfg.top_k < 2:
        raise ValueError("num_acoustic_points and top_k must be at least 2")
    return cfg


def compute_audio_training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: SoftTrainingConfig,
) -> torch.Tensor:
    if cfg.audio_loss_type == "stft_log_l1":
        return stft_magnitude_loss(pred, target)
    if cfg.audio_loss_type == "audiogs_mono_diff":
        return audiogs_mono_diff_loss(
            pred,
            target,
            diff_weight=cfg.audio_diff_weight,
            lre_loss_weight=cfg.audio_lre_loss_weight,
        )
    raise ValueError(f"unknown audio_loss_type {cfg.audio_loss_type!r}")


def compute_total_soft_loss(
    cfg: SoftTrainingConfig,
    acoustic_field: AcousticGaussianField,
    acoustic_state: dict[str, torch.Tensor],
    visual_state: dict[str, torch.Tensor],
    audio_loss: torch.Tensor,
    rgb_loss: torch.Tensor,
    coupling_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    coupling = soft_coupling_losses(acoustic_field, acoustic_state, visual_state)
    coupling_scale = float(coupling_scale)
    total = (
        cfg.audio_weight * audio_loss
        + cfg.rgb_weight * rgb_loss
        + coupling_scale * cfg.anchor_weight * coupling["anchor"]
        + coupling_scale * cfg.motion_weight * coupling["motion"]
        + coupling_scale * cfg.activity_weight * coupling["activity"]
        + cfg.sparse_weight * coupling["sparse"]
    )
    return {
        "total": total,
        "audio": audio_loss.detach(),
        "rgb": rgb_loss.detach(),
        "anchor": coupling["anchor"].detach(),
        "motion": coupling["motion"].detach(),
        "activity": coupling["activity"].detach(),
        "sparse": coupling["sparse"].detach(),
        "coupling_scale": audio_loss.new_tensor(coupling_scale),
    }


def coupling_scale_for_step(step: int, acoustic_steps: int, joint_steps: int) -> float:
    if step < acoustic_steps:
        return 0.0
    if joint_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, (step - acoustic_steps + 1) / float(joint_steps)))


def save_soft_checkpoint(
    output_path: str | Path,
    acoustic_field: AcousticGaussianField,
    audio_renderer: FrequencyTransferRenderer,
    ftgspp_checkpoint: str | Path,
    config: SoftTrainingConfig,
    loss_history: list[dict[str, float]],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "route": "C_soft_av_gaussians",
            "renderer_type": "frequency_transfer",
            "ftgspp_checkpoint": str(ftgspp_checkpoint),
            "acoustic_field": acoustic_field.state_dict(),
            "audio_renderer": {
                "n_fft": audio_renderer.n_fft,
                "hop_length": audio_renderer.hop_length,
                "win_length": audio_renderer.win_length,
                "use_phase_delay": audio_renderer.use_phase_delay,
            },
            "config": asdict(config),
            "loss_history": loss_history,
        },
        output_path,
    )


def train_soft_av_gaussians(
    cfg: SoftTrainingConfig,
    frame_reader: FrameReader | None = None,
) -> list[dict[str, float]]:
    bridge = FTGSRendererBridge.load_checkpoint(cfg.ftgspp_checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(bridge.gaussians.parameters()).device
    with torch.no_grad():
        visual0 = bridge.query_state(torch.zeros(1, 1, device=device))
    acoustic_field = AcousticGaussianField.from_visual_state(
        visual0,
        num_points=cfg.num_acoustic_points,
        anchored_fraction=cfg.anchored_fraction,
        dynamic_fraction=cfg.dynamic_fraction,
    ).to(device)
    audio_renderer = FrequencyTransferRenderer().to(device)
    for parameter in bridge.gaussians.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(
        acoustic_field.parameters(),
        lr=cfg.acoustic_lr,
    )
    visual_dataset = VisualFrameDataset(
        cfg.manifest,
        split="train",
        scale=cfg.visual_scale,
        memmap_root=cfg.ftgspp_memmap,
        frame_reader=frame_reader,
    )
    cropper = TimedAudioCropper(
        cfg.manifest,
        crop_seconds=cfg.audio_window_seconds,
        mode=cfg.audio_crop_mode,
        allow_padding=False,
        bandpass=cfg.audio_bandpass,
    )
    loss_history: list[dict[str, float]] = []
    steps = cfg.acoustic_steps + cfg.joint_steps
    cursor = 0
    for step in range(steps):
        visual_sample = None
        audio_sample = None
        for _attempt in range(len(visual_dataset)):
            candidate = visual_dataset[cursor % len(visual_dataset)]
            cursor += 1
            try:
                audio_sample = cropper.get_crop(str(candidate["camera"]), candidate["time"])
                visual_sample = candidate
                break
            except ValueError as error:
                if "padding" not in str(error):
                    raise
        if visual_sample is None or audio_sample is None:
            raise ValueError("no non-padding Route C samples available")
        visual_sample = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in visual_sample.items()
        }
        render_time = visual_sample["time"]
        visual_state = bridge.query_state(render_time)
        acoustic_state = acoustic_field.query(render_time)
        pred_audio = audio_renderer(
            acoustic_state,
            audio_sample["source_audio"].to(device),
            camera_w2c=visual_sample["w2c"],
        )
        audio_loss = compute_audio_training_loss(pred_audio, audio_sample["target_audio"].to(pred_audio), cfg)
        rgb_loss = audio_loss.new_zeros(())
        optimizer.zero_grad(set_to_none=True)
        losses = compute_total_soft_loss(
            cfg,
            acoustic_field,
            acoustic_state,
            visual_state,
            audio_loss,
            rgb_loss,
            coupling_scale=coupling_scale_for_step(step, cfg.acoustic_steps, cfg.joint_steps),
        )
        losses["total"].backward()
        optimizer.step()
        loss_history.append({key: float(value.detach().cpu().item()) for key, value in losses.items()})
    save_soft_checkpoint(cfg.output, acoustic_field, audio_renderer, cfg.ftgspp_checkpoint, cfg, loss_history)
    return loss_history


def main(argv: Sequence[str] | None = None) -> None:
    cfg = resolve_soft_training_config(build_arg_parser().parse_args(argv))
    history = train_soft_av_gaussians(cfg)
    print({"route": "C_soft_av_gaussians", "steps": len(history), "final": history[-1] if history else None})


if __name__ == "__main__":
    main()
