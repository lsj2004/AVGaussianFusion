import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from avfusion.eval.fair_baseline_comparison import (
    build_fair_comparison_rows,
    compute_valid_aligned_samples,
    verify_protocol,
    write_fair_comparison_tables,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _write_manifest(path: Path, audio_root: Path):
    sample_rate = 16000
    audio_root.mkdir(parents=True, exist_ok=True)
    silence = np.zeros((sample_rate, 2), dtype=np.float32)
    sf.write(audio_root / "near.wav", silence, sample_rate)
    for idx in range(39):
        sf.write(audio_root / f"cam{idx:02d}.wav", silence, sample_rate)
    cameras = {
        f"cam{idx:02d}": {
            "name": f"cam{idx:02d}",
            "index": idx,
            "video_path": f"/tmp/cam{idx:02d}.mp4",
            "audio_path": str(audio_root / f"cam{idx:02d}.wav"),
        }
        for idx in range(39)
    }
    _write_json(
        path,
        {
            "scene_id": "fake_scene",
            "visual_root": "/tmp/visual",
            "audio_root": str(audio_root),
            "fps": 10.0,
            "num_frames": 10,
            "frame_times": [idx / 10.0 for idx in range(10)],
            "cameras": cameras,
            "train_cameras": [f"cam{idx:02d}" for idx in range(39) if idx != 10],
            "eval_cameras": ["cam10"],
            "audio": {
                "sample_rate": sample_rate,
                "channels": 2,
                "crop_seconds": 3.0,
                "crop_samples": sample_rate * 3,
                "source_path": str(audio_root / "near.wav"),
            },
        },
    )


def test_verify_protocol_accepts_allcams_cam10_near_split(tmp_path):
    manifest = tmp_path / "scene_manifest.json"
    _write_manifest(manifest, tmp_path / "audio")
    ftgspp_config = tmp_path / "scene.toml"
    ftgspp_config.write_text(
        "eval_cameras = [10]\n"
        "train_cameras = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]\n"
    )

    report = verify_protocol(
        manifest_path=manifest,
        ftgspp_config_path=ftgspp_config,
        audiogs_summary_path=None,
    )

    assert report["total_cams"] == 39
    assert report["train_cams"] == 38
    assert report["eval_cam"] == "cam10"
    assert report["source"] == "near.wav"
    assert report["ftgspp_train_cams"] == 38
    assert report["ok"] is True


def test_compute_valid_aligned_samples_skips_padding_frames(tmp_path):
    manifest = tmp_path / "scene_manifest.json"
    _write_manifest(manifest, tmp_path / "audio")

    result = compute_valid_aligned_samples(manifest, audio_window_seconds=0.5)

    assert result["valid_frames_per_camera"] == 5
    assert result["train_cams"] == 38
    assert result["valid_aligned_samples"] == 190


def test_build_fair_comparison_rows_includes_required_methods_and_protocol_columns(tmp_path):
    root = tmp_path / "AVGaussianFusion"
    route_a = root / "runs/scene1_opera_a"
    _write_manifest(route_a / "scene_manifest.json", tmp_path / "audio")
    _write_json(
        route_a / "eval_avcloud/audio_summary.json",
        {"camera": "cam10", "MAG": 0.1, "ENV": 0.2, "LRE": 0.3, "DPAM": 0.4, "RTE": 0.5},
    )
    _write_json(
        route_a / "ftgspp/summary.json",
        {"PSNR_mean": {"total": 25.0}, "MSE": {"total": 0.01}, "L1": {"total": 0.02}},
    )
    for run_name, mag in [
        ("scene1_opera_b_joint_av_fair", 0.21),
        ("scene1_opera_b_joint_av_no_warmup_fair", 0.22),
    ]:
        _write_json(
            root / f"runs/{run_name}/eval/audio_summary.json",
            {
                "camera": "cam10",
                "MAG": mag,
                "ENV": mag + 0.1,
                "LRE": mag + 0.2,
                "DPAM": 0.23,
                "DPAM_available": True,
                "dpam_num_windows": 16,
                "RTE": None,
                "num_windows": 10,
                "audio_window_seconds": 0.5,
            },
        )
        _write_json(
            root / f"runs/{run_name}/eval/visual_summary.json",
            {"camera": "cam10", "PSNR": 26.0, "MSE": 0.005, "L1": 0.01, "num_frames": 10},
        )
        _write_json(
            root / f"runs/{run_name}/train_summary.json",
            {"joint_steps": 228, "warmup_steps": 1000 if "no_warmup" not in run_name else 0},
        )
    strict_summary = tmp_path / "strict.json"
    _write_json(
        strict_summary,
        [
            {
                "dataset": "scene1_opera",
                "test_viewpoint": 11,
                "num_frames": 1,
                "overall_metrics": {"MAG": 0.09, "ENV": 0.19, "LRE": 0.29, "DPAM": 0.39, "RTE": 0.49},
            }
        ],
    )

    rows = build_fair_comparison_rows(
        root=root,
        datasets=[
            {
                "dataset": "scene1_opera",
                "route_a_run": "scene1_opera_a",
                "route_b_run": "scene1_opera_b_joint_av_fair",
                "route_b_no_warmup_run": "scene1_opera_b_joint_av_no_warmup_fair",
                "ftgspp_scene": "scene1_opera",
                "strict_audiogs_summary": str(strict_summary),
                "strict_audiogs_artifact": str(tmp_path),
            }
        ],
    )

    methods = {(row["dataset"], row["method"]) for row in rows}
    assert ("scene1_opera", "Separate AudioGS baseline (source code)") in methods
    assert ("scene1_opera", "Separate FreeTimeGS++ baseline (visual only)") in methods
    assert ("scene1_opera", "AVFusion joint warmup") in methods
    assert ("scene1_opera", "AVFusion joint no warmup") in methods
    assert ("scene1_opera", "AVFusion Route A frozen visual + audio head") in methods
    assert all(row["train_cams"] == 38 for row in rows)
    assert all(row["test_cam"] == "cam10" for row in rows)
    assert all(row["source"] == "near.wav" for row in rows)
    assert rows[0]["audio_eval_protocol"]
    joint_rows = [row for row in rows if str(row["method"]).startswith("AVFusion joint")]
    assert all(row["dpam_protocol"] == "sampled 16 visual-frame windows" for row in joint_rows)

    out = tmp_path / "tables"
    write_fair_comparison_tables(rows, out)
    assert len(json.loads((out / "comparison.json").read_text())) == len(rows)
    assert len(list(csv.DictReader((out / "comparison.csv").open()))) == len(rows)
    assert "audio_eval_protocol" in (out / "comparison.md").read_text()
