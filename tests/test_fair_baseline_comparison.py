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
        route_a / "eval_fulltrack/full_audio_summary.json",
        {
            "camera": "cam10",
            "eval_type": "fulltrack",
            "protocol": "source_to_heldout_fulltrack",
            "MAG": 0.11,
            "ENV": 0.21,
            "LRE": 0.31,
            "DPAM": None,
            "DPAM_available": False,
            "RTE": None,
        },
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
        _write_json(
            root / f"runs/{run_name}/eval_fulltrack_audiogs_3s_nonoverlap/full_audio_summary.json",
            {
                "route": "B_joint_av",
                "eval_type": "fulltrack",
                "protocol": "audiogs_3s_nonoverlap_fulltrack",
                "camera": "cam10",
                "MAG": mag + 0.01,
                "ENV": mag + 0.11,
                "LRE": mag + 0.21,
                "DPAM": None,
                "DPAM_available": False,
                "RTE": None,
                "num_windows": 10,
                "audio_window_seconds": 3.0,
            },
        )
    for out_dir, protocol, mag in [
        ("eval_fulltrack_audiogs_3s_nonoverlap", "audiogs_3s_nonoverlap_fulltrack", 0.31),
        ("eval_fulltrack_visual_center_0p5s_ola", "visual_center_overlap_add_fulltrack", 0.32),
    ]:
        _write_json(
            root / f"runs/scene1_opera_c_soft_av_gaussians/{out_dir}/full_audio_summary.json",
            {
                "route": "C_soft_av_gaussians",
                "eval_type": "fulltrack",
                "protocol": protocol,
                "camera": "cam10",
                "MAG": mag,
                "ENV": mag + 0.1,
                "LRE": mag + 0.2,
                "DPAM": None,
                "DPAM_available": False,
                "RTE": None,
                "num_windows": 10,
                "audio_window_seconds": 3.0 if "3s" in out_dir else 0.5,
            },
        )
    _write_json(
        root / "runs/scene1_opera_c_soft_av_gaussians/train_summary.json",
        {"joint_steps": 228, "training_steps": 1228, "route": "C_soft_av_gaussians"},
    )
    for out_dir, protocol, mag in [
        ("eval_fulltrack_audiogs_3s_nonoverlap", "audiogs_3s_nonoverlap_fulltrack", 0.30),
        ("eval_fulltrack_visual_center_0p5s_ola", "visual_center_overlap_add_fulltrack", 0.33),
    ]:
        _write_json(
            root
            / f"runs/scene1_opera_c_soft_av_gaussians_stereo_reg/{out_dir}/full_audio_summary.json",
            {
                "route": "C_soft_av_gaussians",
                "eval_type": "fulltrack",
                "protocol": protocol,
                "camera": "cam10",
                "MAG": mag,
                "ENV": mag + 0.1,
                "LRE": mag + 0.2,
                "DPAM": None,
                "DPAM_available": False,
                "RTE": None,
                "num_windows": 10,
                "audio_window_seconds": 3.0 if "3s" in out_dir else 0.5,
            },
        )
    _write_json(
        root / "runs/scene1_opera_c_soft_av_gaussians_stereo_reg/train_summary.json",
        {"joint_steps": 228, "training_steps": 1228, "route": "C_soft_av_gaussians"},
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
    strict_fulltrack_summary = tmp_path / "strict_fulltrack.json"
    _write_json(
        strict_fulltrack_summary,
        {
            "dataset": "scene1_opera",
            "MAG": 0.08,
            "ENV": 0.18,
            "LRE": 0.28,
            "DPAM": None,
            "RTE": None,
            "num_windows": 1,
            "protocol": "audiogs_3s_nonoverlap_fulltrack",
        },
    )

    rows = build_fair_comparison_rows(
        root=root,
        datasets=[
            {
                "dataset": "scene1_opera",
                "route_a_run": "scene1_opera_a",
                "route_b_run": "scene1_opera_b_joint_av_fair",
                "route_b_no_warmup_run": "scene1_opera_b_joint_av_no_warmup_fair",
                "route_c_run": "scene1_opera_c_soft_av_gaussians",
                "route_c_stereo_reg_run": "scene1_opera_c_soft_av_gaussians_stereo_reg",
                "ftgspp_scene": "scene1_opera",
                "strict_audiogs_summary": str(strict_summary),
                "strict_audiogs_fulltrack_summary": str(strict_fulltrack_summary),
                "strict_audiogs_artifact": str(tmp_path),
            }
        ],
    )

    methods = {(row["dataset"], row["method"]) for row in rows}
    assert ("scene1_opera", "Separate AudioGS baseline (source code)") in methods
    assert ("scene1_opera", "Separate AudioGS baseline (source code full-track)") in methods
    assert ("scene1_opera", "Separate FreeTimeGS++ baseline (visual only)") in methods
    assert ("scene1_opera", "AVFusion joint warmup") in methods
    assert ("scene1_opera", "AVFusion joint no warmup") in methods
    assert (
        "scene1_opera",
        "AVFusion joint warmup (AVFusion full-track AudioGS-style 3s non-overlap)",
    ) in methods
    assert (
        "scene1_opera",
        "AVFusion joint no warmup (AVFusion full-track AudioGS-style 3s non-overlap)",
    ) in methods
    assert ("scene1_opera", "AVFusion Route A frozen visual + audio head") in methods
    assert (
        "scene1_opera",
        "AVFusion Route C soft acoustic Gaussians (AVFusion full-track AudioGS-style 3s non-overlap)",
    ) in methods
    assert (
        "scene1_opera",
        "AVFusion Route C soft acoustic Gaussians (AVFusion full-track visual-center 0.5s overlap-add)",
    ) in methods
    assert (
        "scene1_opera",
        "AVFusion Route C soft acoustic Gaussians + stereo regularization (AVFusion full-track AudioGS-style 3s non-overlap)",
    ) in methods
    assert all(row["train_cams"] == 38 for row in rows)
    assert all(row["test_cam"] == "cam10" for row in rows)
    assert all(row["source"] == "near.wav" for row in rows)
    assert rows[0]["audio_eval_protocol"]
    strict_fulltrack_row = next(
        row for row in rows if row["method"] == "Separate AudioGS baseline (source code full-track)"
    )
    assert strict_fulltrack_row["audio_eval_protocol"] == "AudioGS source-code 3s non-overlap full-track"
    assert strict_fulltrack_row["metric_scope"] == "full_track"
    assert strict_fulltrack_row["MAG"] == 0.08
    route_a_row = next(row for row in rows if row["method"] == "AVFusion Route A frozen visual + audio head")
    assert route_a_row["audio_eval_protocol"] == "AVFusion full-track source-to-heldout audio"
    assert route_a_row["metric_scope"] == "full_track"
    assert route_a_row["dpam_protocol"] == "not_available"
    assert route_a_row["MAG"] == 0.11
    joint_rows = [row for row in rows if str(row["method"]).startswith("AVFusion joint")]
    window_joint_rows = [row for row in joint_rows if row["metric_scope"] == "window_average"]
    fulltrack_joint_rows = [row for row in joint_rows if row["metric_scope"] == "full_track"]
    assert all(row["dpam_protocol"] == "sampled 16 visual-frame windows" for row in window_joint_rows)
    assert len(window_joint_rows) == 2
    assert len(fulltrack_joint_rows) == 2
    assert all(
        row["audio_eval_protocol"] == "AVFusion full-track AudioGS-style 3s non-overlap"
        for row in fulltrack_joint_rows
    )
    route_c = next(
        row
        for row in rows
        if row["method"]
        == "AVFusion Route C soft acoustic Gaussians (AVFusion full-track AudioGS-style 3s non-overlap)"
    )
    assert route_c["audio_eval_protocol"] == "AVFusion full-track AudioGS-style 3s non-overlap"
    assert route_c["metric_scope"] == "full_track"
    assert route_c["dpam_protocol"] == "not_available"
    assert route_c["visual_eval_frames"] == 10
    assert route_c["PSNR"] == 25.0
    assert route_c["steps_or_clips"] == 1228
    route_c_stereo_reg = next(
        row
        for row in rows
        if row["method"]
        == "AVFusion Route C soft acoustic Gaussians + stereo regularization (AVFusion full-track AudioGS-style 3s non-overlap)"
    )
    assert route_c_stereo_reg["audio_eval_protocol"] == "AVFusion full-track AudioGS-style 3s non-overlap"
    assert route_c_stereo_reg["metric_scope"] == "full_track"
    assert route_c_stereo_reg["dpam_protocol"] == "not_available"
    assert route_c_stereo_reg["MAG"] == 0.30
    assert route_c_stereo_reg["steps_or_clips"] == 1228
    route_c_fulltrack_rows = [
        row
        for row in rows
        if str(row["method"]).startswith("AVFusion Route C soft acoustic Gaussians")
    ]
    assert len(route_c_fulltrack_rows) == 4

    out = tmp_path / "tables"
    write_fair_comparison_tables(rows, out)
    assert len(json.loads((out / "comparison.json").read_text())) == len(rows)
    csv_rows = list(csv.DictReader((out / "comparison.csv").open()))
    assert len(csv_rows) == len(rows)
    assert "metric_scope" in csv_rows[0]
    comparison_md = (out / "comparison.md").read_text()
    assert "audio_eval_protocol" in comparison_md
    assert "metric_scope" in comparison_md
