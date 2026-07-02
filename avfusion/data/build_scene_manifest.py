from __future__ import annotations

import argparse
import json
from pathlib import Path

from avfusion.data.manifest import AudioSpec, CameraRecord, SceneManifest


def _camera_name_from_path(path: Path) -> str:
    return path.stem


def _sorted_camera_names(root: Path, suffix: str) -> list[str]:
    return sorted(_camera_name_from_path(p) for p in root.glob(f"cam*{suffix}"))


def build_manifest(
    scene_id: str,
    visual_root: str | Path,
    audio_root: str | Path,
    heldout_camera: str = "cam10",
    fps: float = 30.0,
    num_frames: int = 150,
    sample_rate: int = 16000,
    crop_seconds: float = 3.0,
    output_path: str | Path | None = None,
) -> SceneManifest:
    visual_root = Path(visual_root)
    audio_root = Path(audio_root)
    aligned_audio_root = audio_root / "aligned_16k_stereo"

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
            channels=2,
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_manifest(
        scene_id=args.scene_id,
        visual_root=args.visual_root,
        audio_root=args.audio_root,
        heldout_camera=args.heldout_camera,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
