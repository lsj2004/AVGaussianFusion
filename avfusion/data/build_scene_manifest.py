from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from avfusion.data.manifest import AudioSpec, CameraRecord, SceneManifest


def _camera_name_from_path(path: Path) -> str:
    return path.stem


def _sorted_camera_names(root: Path, suffix: str) -> list[str]:
    return sorted(_camera_name_from_path(p) for p in root.glob(f"cam*{suffix}"))


def _load_visual_metadata(visual_root: Path) -> dict[str, object]:
    manifest_path = visual_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def _resolve_metadata_value(
    name: str,
    requested: float | int | None,
    discovered: object,
) -> float | int:
    if discovered is None:
        if requested is None:
            raise ValueError(
                f"{name} must be provided when visual manifest metadata is absent"
            )
        return requested
    if requested is not None and requested != discovered:
        raise ValueError(
            f"{name} mismatch: requested={requested!r}, visual_manifest={discovered!r}"
        )
    return discovered


def _resolve_audio_value(
    name: str,
    requested: int | None,
    discovered: int,
) -> int:
    if requested is not None and requested != discovered:
        raise ValueError(f"{name} mismatch: requested={requested!r}, wav={discovered!r}")
    return discovered


def _resolve_num_frames(
    requested: int | None,
    discovered: object,
) -> int:
    if discovered is None:
        if requested is None:
            raise ValueError(
                "num_frames must be provided when visual manifest metadata is absent"
            )
        return int(requested)

    discovered_int = int(discovered)
    if requested is None:
        return discovered_int
    requested_int = int(requested)
    if requested_int > discovered_int:
        raise ValueError(
            "num_frames exceeds visual manifest: "
            f"requested={requested_int!r}, visual_manifest={discovered_int!r}"
        )
    return requested_int


def _audio_info(path: Path) -> tuple[int, int]:
    info = sf.info(path)
    return int(info.samplerate), int(info.channels)


def build_manifest(
    scene_id: str,
    visual_root: str | Path,
    audio_root: str | Path,
    heldout_camera: str = "cam10",
    fps: float | None = None,
    num_frames: int | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    crop_seconds: float = 3.0,
    output_path: str | Path | None = None,
) -> SceneManifest:
    visual_root = Path(visual_root)
    audio_root = Path(audio_root)
    aligned_audio_root = audio_root / "aligned_16k_stereo"
    visual_metadata = _load_visual_metadata(visual_root)

    video_names = _sorted_camera_names(visual_root, ".mp4")
    audio_names = _sorted_camera_names(aligned_audio_root, ".wav")
    if video_names != audio_names:
        missing_audio = sorted(set(video_names) - set(audio_names))
        missing_video = sorted(set(audio_names) - set(video_names))
        raise ValueError(
            f"camera mismatch: missing_audio={missing_audio}, missing_video={missing_video}"
        )
    if heldout_camera not in video_names:
        raise ValueError(f"heldout camera {heldout_camera!r} is not present")

    source_path = aligned_audio_root / "near.wav"
    if not source_path.exists():
        raise FileNotFoundError(f"missing source audio: {source_path}")

    discovered_sample_rate, discovered_channels = _audio_info(source_path)
    sample_rate = _resolve_audio_value(
        "sample_rate", sample_rate, discovered_sample_rate
    )
    channels = _resolve_audio_value("channels", channels, discovered_channels)
    for name in audio_names:
        audio_sample_rate, audio_channels = _audio_info(aligned_audio_root / f"{name}.wav")
        if audio_sample_rate != sample_rate or audio_channels != channels:
            raise ValueError(
                f"audio metadata mismatch for {name}: "
                f"sample_rate={audio_sample_rate}, channels={audio_channels}"
            )

    fps = float(_resolve_metadata_value("fps", fps, visual_metadata.get("fps")))
    num_frames = _resolve_num_frames(num_frames, visual_metadata.get("num_frames"))
    manifest_camera_names = visual_metadata.get("camera_names")
    if manifest_camera_names is not None and list(manifest_camera_names) != video_names:
        raise ValueError("camera_names in visual manifest do not match cam*.mp4 files")

    cameras = {
        name: CameraRecord(
            name=name,
            index=int(name.replace("cam", "")),
            video_path=str(visual_root / f"{name}.mp4"),
            audio_path=str(aligned_audio_root / f"{name}.wav"),
        )
        for name in video_names
    }
    train_cameras = [name for name in video_names if name != heldout_camera]
    frame_times = [idx / fps for idx in range(num_frames)]
    manifest = SceneManifest(
        scene_id=scene_id,
        visual_root=str(visual_root),
        audio_root=str(audio_root),
        fps=float(fps),
        num_frames=int(num_frames),
        frame_times=frame_times,
        cameras=cameras,
        train_cameras=train_cameras,
        eval_cameras=[heldout_camera],
        audio=AudioSpec(
            sample_rate=int(sample_rate),
            channels=int(channels),
            crop_seconds=float(crop_seconds),
            crop_samples=int(round(sample_rate * crop_seconds)),
            source_path=str(source_path),
        ),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest.to_dict(), indent=2))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default="scene1_opera")
    parser.add_argument("--visual-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--heldout-camera", default="cam10")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--channels", type=int)
    parser.add_argument("--crop-seconds", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_manifest(
        scene_id=args.scene_id,
        visual_root=args.visual_root,
        audio_root=args.audio_root,
        heldout_camera=args.heldout_camera,
        fps=args.fps,
        num_frames=args.num_frames,
        sample_rate=args.sample_rate,
        channels=args.channels,
        crop_seconds=args.crop_seconds,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
