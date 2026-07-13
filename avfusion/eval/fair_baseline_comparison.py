from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import soundfile as sf

from avfusion.data.manifest import SceneManifest


FAIR_COLUMNS = [
    "dataset",
    "method",
    "train_cams",
    "test_cam",
    "source",
    "steps_or_clips",
    "audio_eval_protocol",
    "metric_scope",
    "dpam_protocol",
    "visual_eval_frames",
    "MAG",
    "ENV",
    "LRE",
    "DPAM",
    "RTE",
    "PSNR",
    "MSE",
    "L1",
    "artifact",
]


DEFAULT_DATASETS = [
    {
        "dataset": "scene1_opera",
        "route_a_run": "scene1_opera_a",
        "route_b_run": "scene1_opera_b_joint_av_fair",
        "route_b_no_warmup_run": "scene1_opera_b_joint_av_no_warmup_fair",
        "route_b_spectral_no_warmup_run": "scene1_opera_b_joint_av_spectral_no_warmup",
        "route_b_spectral_strict_audiogs_run": "scene1_opera_b_joint_av_spectral_no_warmup_audiogs_strict",
        "route_c_run": "scene1_opera_c_soft_av_gaussians",
        "route_c_stereo_reg_run": "scene1_opera_c_soft_av_gaussians_stereo_reg",
        "ftgspp_scene": "scene1_opera",
        "ftgspp_config": "configs/ftgspp_scene1_opera/scene1_opera.toml",
        "strict_audiogs_summary": "/mnt/sda/lisujing/Dataset/audioGS-replay/runs/strict_audiogs_cam10_allcams/strict_audiogs_cam10_allcams_summary.json",
        "strict_audiogs_fulltrack_summary": "/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/fair_baseline_comparison/audiogs_fulltrack_baseline/scene1_opera/full_audio_summary.json",
        "strict_audiogs_artifact": "/mnt/sda/lisujing/Dataset/audioGS-replay/runs/strict_audiogs_cam10_allcams/eval/SC-scene1-opera-cam10-allcams/viewpoint_11",
    },
    {
        "dataset": "scene7_playing_300",
        "route_a_run": "scene7_playing_300_a",
        "route_b_run": "scene7_playing_300_b_joint_av_fair",
        "route_b_no_warmup_run": "scene7_playing_300_b_joint_av_no_warmup_fair",
        "route_b_spectral_no_warmup_run": "scene7_playing_300_b_joint_av_spectral_no_warmup",
        "route_b_spectral_strict_audiogs_run": "scene7_playing_300_b_joint_av_spectral_no_warmup_audiogs_strict",
        "route_c_run": "scene7_playing_300_c_soft_av_gaussians",
        "route_c_stereo_reg_run": "scene7_playing_300_c_soft_av_gaussians_stereo_reg",
        "ftgspp_scene": "Scene7playing",
        "ftgspp_config": "configs/ftgspp_scene7_playing_300/Scene7playing.toml",
        "strict_audiogs_summary": "/mnt/sda/lisujing/Dataset/audioGS-replay/runs/strict_audiogs_cam10_allcams/strict_audiogs_cam10_allcams_summary.json",
        "strict_audiogs_fulltrack_summary": "/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/fair_baseline_comparison/audiogs_fulltrack_baseline/scene7_playing_300/full_audio_summary.json",
        "strict_audiogs_artifact": "/mnt/sda/lisujing/Dataset/audioGS-replay/runs/strict_audiogs_cam10_allcams/eval/SC-scene7-playing-300-cam10-allcams/viewpoint_11",
    },
]


def _load_json(path: str | Path | None) -> Any:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _metric_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in data:
            continue
        value = data.get(key)
        if isinstance(value, dict):
            return value.get("total")
        return value
    return None


def _parse_camera_list(text: str, key: str) -> list[int] | None:
    match = re.search(rf"{re.escape(key)}\s*=\s*\[([^\]]*)\]", text)
    if match is None:
        return None
    values = []
    for item in match.group(1).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def verify_protocol(
    manifest_path: str | Path,
    ftgspp_config_path: str | Path | None = None,
    audiogs_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = SceneManifest.load(manifest_path)
    source = Path(manifest.audio.source_path).name
    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "total_cams": len(manifest.cameras),
        "train_cams": len(manifest.train_cameras),
        "eval_cameras": list(manifest.eval_cameras),
        "eval_cam": manifest.eval_cameras[0] if manifest.eval_cameras else None,
        "source": source,
        "ok": True,
        "errors": [],
    }
    errors = report["errors"]
    if len(manifest.cameras) != 39:
        errors.append(f"expected 39 total cameras, got {len(manifest.cameras)}")
    if len(manifest.train_cameras) != 38:
        errors.append(f"expected 38 train cameras, got {len(manifest.train_cameras)}")
    if manifest.eval_cameras != ["cam10"]:
        errors.append(f"expected eval_cameras=['cam10'], got {manifest.eval_cameras}")
    if source != "near.wav":
        errors.append(f"expected source near.wav, got {source}")

    if ftgspp_config_path is not None and Path(ftgspp_config_path).exists():
        text = Path(ftgspp_config_path).read_text()
        train_cameras = _parse_camera_list(text, "train_cameras")
        eval_cameras = _parse_camera_list(text, "eval_cameras")
        report["ftgspp_train_cams"] = len(train_cameras) if train_cameras is not None else None
        report["ftgspp_eval_cameras"] = eval_cameras
        if train_cameras is None or len(train_cameras) != 38:
            errors.append("FTGS++ config does not list 38 train cameras")
        if eval_cameras != [10]:
            errors.append(f"FTGS++ config eval_cameras should be [10], got {eval_cameras}")

    audiogs_summary = _load_json(audiogs_summary_path)
    if isinstance(audiogs_summary, list):
        test_viewpoints = sorted(
            {
                int(row["test_viewpoint"])
                for row in audiogs_summary
                if row.get("test_viewpoint") is not None
            }
        )
        report["audiogs_test_viewpoints"] = test_viewpoints
        if test_viewpoints and test_viewpoints != [11]:
            errors.append(f"AudioGS strict summary should use viewpoint 11, got {test_viewpoints}")

    report["ok"] = len(errors) == 0
    return report


def _audio_frames(path: str | Path) -> int:
    return int(sf.info(str(path)).frames)


def compute_valid_aligned_samples(
    manifest_path: str | Path,
    audio_window_seconds: float,
) -> dict[str, Any]:
    manifest = SceneManifest.load(manifest_path)
    sample_rate = int(manifest.audio.sample_rate)
    crop_samples = int(round(float(audio_window_seconds) * sample_rate))
    source_frames = _audio_frames(manifest.audio.source_path)
    per_camera_valid: dict[str, int] = {}
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
        per_camera_valid[camera] = valid
    valid_aligned_samples = sum(per_camera_valid.values())
    unique_valid_counts = sorted(set(per_camera_valid.values()))
    return {
        "manifest": str(manifest_path),
        "audio_window_seconds": float(audio_window_seconds),
        "crop_samples": crop_samples,
        "train_cams": len(manifest.train_cameras),
        "eval_cam": manifest.eval_cameras[0] if manifest.eval_cameras else None,
        "valid_frames_per_camera": unique_valid_counts[0] if len(unique_valid_counts) == 1 else None,
        "valid_frames_by_camera": per_camera_valid,
        "valid_aligned_samples": int(valid_aligned_samples),
        "source_audio": Path(manifest.audio.source_path).name,
    }


def _strict_audiogs_row(summary: Any, dataset: str) -> dict[str, Any]:
    if not isinstance(summary, list):
        return {}
    for row in summary:
        if row.get("dataset") == dataset:
            metrics = row.get("overall_metrics") or row.get("metrics") or {}
            return {**row, **metrics}
    return {}


def _run_summary(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


def _manifest_protocol(root: Path, run_name: str) -> dict[str, Any]:
    manifest = SceneManifest.load(root / "runs" / run_name / "scene_manifest.json")
    return {
        "train_cams": len(manifest.train_cameras),
        "test_cam": manifest.eval_cameras[0] if manifest.eval_cameras else None,
        "source": Path(manifest.audio.source_path).name,
        "num_frames": int(manifest.num_frames),
    }


def _artifact(path: str | Path) -> str:
    return str(path)


def _joint_dpam_protocol(audio_summary: dict[str, Any]) -> str:
    explicit = audio_summary.get("dpam_protocol")
    if explicit:
        return str(explicit)
    if audio_summary.get("DPAM_available"):
        num_windows = audio_summary.get("dpam_num_windows")
        if num_windows is not None:
            return f"sampled {num_windows} visual-frame windows"
        return "sampled visual-frame windows"
    return "not_available"


def _audio_window_protocol(audio_summary: dict[str, Any]) -> str:
    protocol = audio_summary.get("audio_eval_protocol")
    if protocol == "visual_center":
        return "AVFusion timed visual-frame audio window average"
    if protocol == "visual_center_skip_padding":
        return "AVFusion timed visual-frame audio window average (skip padding)"
    if protocol == "audiogs_start_nonoverlap":
        return "AVFusion AudioGS-style non-overlap audio window average"
    return "AVFusion timed visual-frame audio window average"


def _fulltrack_audio_protocol(audio_summary: dict[str, Any]) -> str:
    protocol = audio_summary.get("protocol")
    if protocol == "source_to_heldout_fulltrack":
        return "AVFusion full-track source-to-heldout audio"
    if protocol == "audiogs_3s_nonoverlap_fulltrack":
        return "AVFusion full-track AudioGS-style 3s non-overlap"
    if protocol == "visual_center_overlap_add_fulltrack":
        return "AVFusion full-track visual-center 0.5s overlap-add"
    return _audio_window_protocol(audio_summary)


def _route_c_fulltrack_rows(
    base: dict[str, Any],
    root: Path,
    run_name: str,
    method_prefix: str,
    train_summary: dict[str, Any],
    visual_summary: dict[str, Any],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    run_dir = root / "runs" / run_name
    summaries = [
        run_dir / "eval_fulltrack_audiogs_3s_nonoverlap" / "full_audio_summary.json",
        run_dir / "eval_fulltrack_visual_center_0p5s_ola" / "full_audio_summary.json",
    ]
    rows = []
    for summary_path in summaries:
        audio_summary = _run_summary(summary_path)
        if not audio_summary:
            continue
        rows.append(
            {
                **base,
                "method": f"{method_prefix} ({_fulltrack_audio_protocol(audio_summary)})",
                "steps_or_clips": train_summary.get("training_steps") or train_summary.get("joint_steps"),
                "audio_eval_protocol": _fulltrack_audio_protocol(audio_summary),
                "metric_scope": "full_track",
                "dpam_protocol": _joint_dpam_protocol(audio_summary),
                "visual_eval_frames": visual_summary.get("num_frames") or protocol["num_frames"],
                "MAG": audio_summary.get("MAG"),
                "ENV": audio_summary.get("ENV"),
                "LRE": audio_summary.get("LRE"),
                "DPAM": audio_summary.get("DPAM"),
                "RTE": audio_summary.get("RTE"),
                "PSNR": _metric_value(visual_summary, "PSNR_mean", "PSNR"),
                "MSE": _metric_value(visual_summary, "MSE_mean", "MSE"),
                "L1": _metric_value(visual_summary, "L1_mean", "L1"),
                "artifact": _artifact(run_dir),
            }
        )
    return rows


def build_fair_comparison_rows(
    root: str | Path,
    datasets: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for spec in datasets or DEFAULT_DATASETS:
        dataset = str(spec["dataset"])
        route_a_run = str(spec["route_a_run"])
        route_b_run = str(spec["route_b_run"])
        route_b_no_warmup_run = str(spec["route_b_no_warmup_run"])
        route_b_spectral_no_warmup_run = spec.get("route_b_spectral_no_warmup_run")
        route_b_spectral_no_warmup_run = (
            str(route_b_spectral_no_warmup_run)
            if route_b_spectral_no_warmup_run is not None
            else None
        )
        route_b_spectral_strict_audiogs_run = spec.get("route_b_spectral_strict_audiogs_run")
        route_b_spectral_strict_audiogs_run = (
            str(route_b_spectral_strict_audiogs_run)
            if route_b_spectral_strict_audiogs_run is not None
            else None
        )
        route_c_run = spec.get("route_c_run")
        route_c_run = str(route_c_run) if route_c_run is not None else None
        route_c_stereo_reg_run = spec.get("route_c_stereo_reg_run")
        route_c_stereo_reg_run = (
            str(route_c_stereo_reg_run) if route_c_stereo_reg_run is not None else None
        )
        protocol = _manifest_protocol(root, route_a_run)
        route_a_audio = _run_summary(root / "runs" / route_a_run / "eval_avcloud" / "audio_summary.json")
        if not route_a_audio:
            route_a_audio = _run_summary(root / "runs" / route_a_run / "eval" / "audio_summary.json")
        route_a_fulltrack_audio = _run_summary(
            root / "runs" / route_a_run / "eval_fulltrack" / "full_audio_summary.json"
        )
        if route_a_fulltrack_audio:
            route_a_audio = route_a_fulltrack_audio
        route_a_visual = _run_summary(root / "runs" / route_a_run / "ftgspp" / "summary.json")
        strict = _strict_audiogs_row(_load_json(spec.get("strict_audiogs_summary")), dataset)
        strict_fulltrack = _run_summary(spec.get("strict_audiogs_fulltrack_summary"))
        route_b_audio = _run_summary(root / "runs" / route_b_run / "eval" / "audio_summary.json")
        route_b_visual = _run_summary(root / "runs" / route_b_run / "eval" / "visual_summary.json")
        route_b_train = _run_summary(root / "runs" / route_b_run / "train_summary.json")
        route_b_nowarm_audio = _run_summary(
            root / "runs" / route_b_no_warmup_run / "eval" / "audio_summary.json"
        )
        route_b_nowarm_visual = _run_summary(
            root / "runs" / route_b_no_warmup_run / "eval" / "visual_summary.json"
        )
        route_b_nowarm_train = _run_summary(root / "runs" / route_b_no_warmup_run / "train_summary.json")
        spectral_audio = {}
        spectral_visual = {}
        spectral_train = {}
        if route_b_spectral_no_warmup_run is not None:
            spectral_audio = _run_summary(
                root / "runs" / route_b_spectral_no_warmup_run / "eval" / "audio_summary.json"
            )
            spectral_visual = _run_summary(
                root / "runs" / route_b_spectral_no_warmup_run / "eval" / "visual_summary.json"
            )
            spectral_train = _run_summary(
                root / "runs" / route_b_spectral_no_warmup_run / "train_summary.json"
            )
        spectral_strict_audio = {}
        spectral_strict_visual = {}
        spectral_strict_train = {}
        if route_b_spectral_strict_audiogs_run is not None:
            spectral_strict_audio = _run_summary(
                root / "runs" / route_b_spectral_strict_audiogs_run / "eval" / "audio_summary.json"
            )
            spectral_strict_visual = _run_summary(
                root / "runs" / route_b_spectral_strict_audiogs_run / "eval" / "visual_summary.json"
            )
            spectral_strict_train = _run_summary(
                root / "runs" / route_b_spectral_strict_audiogs_run / "train_summary.json"
            )
        route_c_visual = route_a_visual
        route_c_train = {}
        if route_c_run is not None:
            route_c_train = _run_summary(root / "runs" / route_c_run / "train_summary.json")
        route_c_stereo_reg_train = {}
        if route_c_stereo_reg_run is not None:
            route_c_stereo_reg_train = _run_summary(
                root / "runs" / route_c_stereo_reg_run / "train_summary.json"
            )

        base = {
            "dataset": dataset,
            "train_cams": protocol["train_cams"],
            "test_cam": protocol["test_cam"],
            "source": protocol["source"],
        }
        rows.extend(
            [
                {
                    **base,
                    "method": "Separate AudioGS baseline (source code)",
                    "steps_or_clips": strict.get("num_frames"),
                    "audio_eval_protocol": "AudioGS source-code 3s non-overlap clip eval",
                    "metric_scope": "clip_average",
                    "dpam_protocol": "AudioGS source eval",
                    "visual_eval_frames": None,
                    "MAG": strict.get("MAG"),
                    "ENV": strict.get("ENV"),
                    "LRE": strict.get("LRE"),
                    "DPAM": strict.get("DPAM"),
                    "RTE": strict.get("RTE"),
                    "PSNR": None,
                    "MSE": None,
                    "L1": None,
                    "artifact": _artifact(spec.get("strict_audiogs_artifact", "")),
                },
            ]
        )
        if strict_fulltrack:
            rows.append(
                {
                    **base,
                    "method": "Separate AudioGS baseline (source code full-track)",
                    "steps_or_clips": strict_fulltrack.get("num_windows"),
                    "audio_eval_protocol": "AudioGS source-code 3s non-overlap full-track",
                    "metric_scope": "full_track",
                    "dpam_protocol": _joint_dpam_protocol(strict_fulltrack),
                    "visual_eval_frames": None,
                    "MAG": strict_fulltrack.get("MAG"),
                    "ENV": strict_fulltrack.get("ENV"),
                    "LRE": strict_fulltrack.get("LRE"),
                    "DPAM": strict_fulltrack.get("DPAM"),
                    "RTE": strict_fulltrack.get("RTE"),
                    "PSNR": None,
                    "MSE": None,
                    "L1": None,
                    "artifact": _artifact(spec.get("strict_audiogs_artifact", "")),
                }
            )
        rows.extend(
            [
                {
                    **base,
                    "method": "Separate FreeTimeGS++ baseline (visual only)",
                    "steps_or_clips": None,
                    "audio_eval_protocol": None,
                    "metric_scope": "visual_only",
                    "dpam_protocol": None,
                    "visual_eval_frames": protocol["num_frames"],
                    "MAG": None,
                    "ENV": None,
                    "LRE": None,
                    "DPAM": None,
                    "RTE": None,
                    "PSNR": _metric_value(route_a_visual, "PSNR_mean", "PSNR"),
                    "MSE": _metric_value(route_a_visual, "MSE_mean", "MSE"),
                    "L1": _metric_value(route_a_visual, "L1_mean", "L1"),
                    "artifact": _artifact(root / "runs" / route_a_run / "ftgspp"),
                },
                {
                    **base,
                    "method": "AVFusion Route A frozen visual + audio head",
                    "steps_or_clips": 1,
                    "audio_eval_protocol": (
                        _fulltrack_audio_protocol(route_a_audio)
                        if route_a_fulltrack_audio
                        else "AVFusion AudioGS-style heldout 3s crop eval"
                    ),
                    "metric_scope": "full_track" if route_a_fulltrack_audio else "clip_average",
                    "dpam_protocol": (
                        _joint_dpam_protocol(route_a_audio)
                        if route_a_fulltrack_audio
                        else "AVCloud/DPAM full crop eval"
                    ),
                    "visual_eval_frames": protocol["num_frames"],
                    "MAG": route_a_audio.get("MAG"),
                    "ENV": route_a_audio.get("ENV"),
                    "LRE": route_a_audio.get("LRE"),
                    "DPAM": route_a_audio.get("DPAM"),
                    "RTE": route_a_audio.get("RTE"),
                    "PSNR": _metric_value(route_a_visual, "PSNR_mean", "PSNR"),
                    "MSE": _metric_value(route_a_visual, "MSE_mean", "MSE"),
                    "L1": _metric_value(route_a_visual, "L1_mean", "L1"),
                    "artifact": _artifact(root / "runs" / route_a_run),
                },
                {
                    **base,
                    "method": "AVFusion joint warmup",
                    "steps_or_clips": route_b_train.get("joint_steps"),
                    "audio_eval_protocol": "AVFusion timed visual-frame audio window average",
                    "metric_scope": "window_average",
                    "dpam_protocol": _joint_dpam_protocol(route_b_audio),
                    "visual_eval_frames": route_b_visual.get("num_frames"),
                    "MAG": route_b_audio.get("MAG"),
                    "ENV": route_b_audio.get("ENV"),
                    "LRE": route_b_audio.get("LRE"),
                    "DPAM": route_b_audio.get("DPAM"),
                    "RTE": route_b_audio.get("RTE"),
                    "PSNR": route_b_visual.get("PSNR"),
                    "MSE": route_b_visual.get("MSE"),
                    "L1": route_b_visual.get("L1"),
                    "artifact": _artifact(root / "runs" / route_b_run),
                },
                {
                    **base,
                    "method": "AVFusion joint no warmup",
                    "steps_or_clips": route_b_nowarm_train.get("joint_steps"),
                    "audio_eval_protocol": "AVFusion timed visual-frame audio window average",
                    "metric_scope": "window_average",
                    "dpam_protocol": _joint_dpam_protocol(route_b_nowarm_audio),
                    "visual_eval_frames": route_b_nowarm_visual.get("num_frames"),
                    "MAG": route_b_nowarm_audio.get("MAG"),
                    "ENV": route_b_nowarm_audio.get("ENV"),
                    "LRE": route_b_nowarm_audio.get("LRE"),
                    "DPAM": route_b_nowarm_audio.get("DPAM"),
                    "RTE": route_b_nowarm_audio.get("RTE"),
                    "PSNR": route_b_nowarm_visual.get("PSNR"),
                    "MSE": route_b_nowarm_visual.get("MSE"),
                    "L1": route_b_nowarm_visual.get("L1"),
                    "artifact": _artifact(root / "runs" / route_b_no_warmup_run),
                },
            ]
        )
        if route_b_spectral_no_warmup_run is not None and (
            root / "runs" / route_b_spectral_no_warmup_run
        ).exists():
            rows.append(
                {
                    **base,
                    "method": "AVFusion spectral audio head no warmup",
                    "steps_or_clips": spectral_train.get("joint_steps"),
                    "audio_eval_protocol": "AVFusion timed visual-frame audio window average",
                    "metric_scope": "window_average",
                    "dpam_protocol": _joint_dpam_protocol(spectral_audio),
                    "visual_eval_frames": spectral_visual.get("num_frames"),
                    "MAG": spectral_audio.get("MAG"),
                    "ENV": spectral_audio.get("ENV"),
                    "LRE": spectral_audio.get("LRE"),
                    "DPAM": spectral_audio.get("DPAM"),
                    "RTE": spectral_audio.get("RTE"),
                    "PSNR": spectral_visual.get("PSNR"),
                    "MSE": spectral_visual.get("MSE"),
                    "L1": spectral_visual.get("L1"),
                    "artifact": _artifact(root / "runs" / route_b_spectral_no_warmup_run),
                }
            )
        if route_b_spectral_strict_audiogs_run is not None and (
            root / "runs" / route_b_spectral_strict_audiogs_run
        ).exists():
            rows.append(
                {
                    **base,
                    "method": "AVFusion spectral audio head no warmup + strict AudioGS audio",
                    "steps_or_clips": spectral_strict_train.get("joint_steps"),
                    "audio_eval_protocol": "AVFusion timed visual-frame audio window average",
                    "metric_scope": "window_average",
                    "dpam_protocol": _joint_dpam_protocol(spectral_strict_audio),
                    "visual_eval_frames": spectral_strict_visual.get("num_frames"),
                    "MAG": spectral_strict_audio.get("MAG"),
                    "ENV": spectral_strict_audio.get("ENV"),
                    "LRE": spectral_strict_audio.get("LRE"),
                    "DPAM": spectral_strict_audio.get("DPAM"),
                    "RTE": spectral_strict_audio.get("RTE"),
                    "PSNR": spectral_strict_visual.get("PSNR"),
                    "MSE": spectral_strict_visual.get("MSE"),
                    "L1": spectral_strict_visual.get("L1"),
                    "artifact": _artifact(root / "runs" / route_b_spectral_strict_audiogs_run),
                }
            )
        if route_c_run is not None and (root / "runs" / route_c_run).exists():
            rows.extend(
                _route_c_fulltrack_rows(
                    base,
                    root,
                    route_c_run,
                    "AVFusion Route C soft acoustic Gaussians",
                    route_c_train,
                    route_c_visual,
                    protocol,
                )
            )
        if route_c_stereo_reg_run is not None and (root / "runs" / route_c_stereo_reg_run).exists():
            rows.extend(
                _route_c_fulltrack_rows(
                    base,
                    root,
                    route_c_stereo_reg_run,
                    "AVFusion Route C soft acoustic Gaussians + stereo regularization",
                    route_c_stereo_reg_train,
                    route_c_visual,
                    protocol,
                )
            )
    return rows


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_fair_comparison_tables(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAIR_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FAIR_COLUMNS})
    lines = [
        "# Fair Baseline Comparison",
        "",
        "Lower is better for MAG/ENV/LRE/DPAM/RTE. Higher is better for PSNR.",
        "Use metric_scope to separate non-comparable audio aggregations: full_track, window_average, clip_average, and visual_only.",
        "",
        "| " + " | ".join(FAIR_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(FAIR_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(key)) for key in FAIR_COLUMNS) + " |")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")


def validate_artifacts(rows: list[dict[str, Any]]) -> list[str]:
    missing = []
    for row in rows:
        artifact = row.get("artifact")
        if artifact and not Path(str(artifact)).exists():
            missing.append(str(artifact))
    return missing


def _safe_link_name(row: dict[str, Any]) -> str:
    raw = f"{row.get('dataset', 'dataset')}__{row.get('method', 'method')}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def link_media_artifacts(rows: list[dict[str, Any]], media_dir: str | Path) -> None:
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for row in rows:
        artifact = row.get("artifact")
        if not artifact:
            continue
        source = Path(str(artifact))
        if not source.exists():
            continue
        link = media_dir / _safe_link_name(row)
        if link.exists() or link.is_symlink():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(source, target_is_directory=source.is_dir())
        index.append({**row, "media_link": str(link), "artifact": str(source)})
    (media_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fair AVFusion vs strict baseline comparison tables.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="AVGaussianFusion repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <root>/runs/fair_baseline_comparison.",
    )
    parser.add_argument("--check-artifacts", action="store_true")
    parser.add_argument("--link-media", action="store_true")
    parser.add_argument("--protocol-report", action="store_true")
    parser.add_argument("--print-valid-joint-steps", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--audio-window-seconds", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.print_valid_joint_steps:
        if not args.manifest:
            raise ValueError("--manifest is required with --print-valid-joint-steps")
        valid = compute_valid_aligned_samples(
            args.manifest,
            audio_window_seconds=args.audio_window_seconds,
        )
        print(int(valid["valid_aligned_samples"]))
        return
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "runs" / "fair_baseline_comparison"
    rows = build_fair_comparison_rows(root)
    write_fair_comparison_tables(rows, output_dir / "metrics")
    if args.link_media:
        link_media_artifacts(rows, output_dir / "media")
    reports = []
    if args.protocol_report:
        for spec in DEFAULT_DATASETS:
            reports.append(
                verify_protocol(
                    manifest_path=root / "runs" / str(spec["route_a_run"]) / "scene_manifest.json",
                    ftgspp_config_path=root / str(spec["ftgspp_config"]),
                    audiogs_summary_path=spec.get("strict_audiogs_summary"),
                )
            )
        (output_dir / "metrics" / "protocol_report.json").write_text(
            json.dumps(reports, indent=2, sort_keys=True) + "\n"
        )
    missing = validate_artifacts(rows) if args.check_artifacts else []
    if missing:
        raise FileNotFoundError("missing artifact paths:\n" + "\n".join(missing))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": len(rows),
                "protocol_reports": reports,
                "missing_artifacts": missing,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
