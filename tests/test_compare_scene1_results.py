import json

from avfusion.eval.compare_scene1_results import build_comparison_rows, write_comparison_tables


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_build_comparison_rows_collects_route_a_and_route_b_metrics(tmp_path):
    _write_json(
        tmp_path / "runs/scene1_opera_a/eval/audio_summary.json",
        {
            "MAG": 0.1,
            "ENV": 0.2,
            "LRE": 0.3,
            "DPAM": 0.4,
            "DPAM_available": True,
            "RTE": None,
            "RTE_available": False,
            "camera": "cam10",
        },
    )
    _write_json(
        tmp_path / "runs/scene1_opera_a/ftgspp/summary.json",
        {
            "PSNR_mean": {"total": 26.68},
            "SSIM-1": {"total": 0.823},
            "LPIPS-Alex": {"total": 0.273},
        },
    )
    for run_name, mag in [
        ("scene1_opera_b_joint_av", 0.22),
        ("scene1_opera_b_joint_av_no_warmup", 0.21),
    ]:
        _write_json(
            tmp_path / f"runs/{run_name}/eval/audio_summary.json",
            {
                "MAG": mag,
                "ENV": mag + 0.1,
                "LRE": mag + 0.2,
                "DPAM": None,
                "DPAM_available": False,
                "RTE": None,
                "RTE_available": False,
                "camera": "cam10",
                "num_windows": 150,
                "audio_window_seconds": 0.5,
            },
        )
        _write_json(
            tmp_path / f"runs/{run_name}/eval/visual_summary.json",
            {"PSNR": 26.7, "camera": "cam10", "num_frames": 150},
        )

    rows = build_comparison_rows(tmp_path)

    assert [row["method"] for row in rows] == [
        "Route A frozen carrier",
        "Route B joint AV",
        "Route B joint AV no warmup",
    ]
    assert rows[0]["DPAM"] == 0.4
    assert rows[0]["PSNR"] == 26.68
    assert rows[1]["MAG"] == 0.22
    assert rows[2]["audio_windows"] == 150


def test_write_comparison_tables_writes_csv_json_and_markdown(tmp_path):
    rows = [
        {
            "method": "Route A frozen carrier",
            "MAG": 0.1,
            "DPAM": None,
        }
    ]

    write_comparison_tables(rows, tmp_path)

    assert (tmp_path / "comparison.json").exists()
    assert "Route A frozen carrier" in (tmp_path / "comparison.csv").read_text()
    assert "N/A" in (tmp_path / "comparison.md").read_text()
