from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/sda/lisujing/Dataset/AVGaussianFusion")
RUNS = {
    "scene1_opera": ROOT / "runs" / "scene1_opera_b_joint_av_audiogs_head_camera_frame",
    "scene7_playing_300": ROOT / "runs" / "scene7_playing_300_b_joint_av_audiogs_head_camera_frame",
}
OUT_JSON = ROOT / "runs" / "phase_mrstft_metrics_summary.json"
OUT_MD = ROOT / "runs" / "phase_mrstft_metrics_summary.md"


def _load(path: Path) -> dict:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text())


def _metric(row: dict, key: str) -> str:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _first_metric(row: dict, *keys: str):
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def main() -> None:
    rows = []
    for dataset, run_dir in RUNS.items():
        audio = _load(run_dir / "eval" / "audio_summary.json")
        visual = _load(run_dir / "eval" / "visual_summary.json")
        train = _load(run_dir / "train_summary.json")
        rows.append(
            {
                "dataset": dataset,
                "run_dir": str(run_dir),
                "joint_steps": train.get("joint_steps"),
                "audio_window_seconds": audio.get("audio_window_seconds", train.get("audio_window_seconds")),
                "MAG": audio.get("MAG"),
                "ENV": audio.get("ENV"),
                "LRE": audio.get("LRE"),
                "DPAM": audio.get("DPAM"),
                "DPAM_available": audio.get("DPAM_available"),
                "DPAM_error": audio.get("DPAM_error"),
                "RTE": audio.get("RTE"),
                "RTE_available": audio.get("RTE_available"),
                "PSNR": _first_metric(visual, "PSNR", "psnr"),
                "MSE": _first_metric(visual, "MSE", "mse"),
                "L1": _first_metric(visual, "L1", "l1"),
                "audio_summary": str(run_dir / "eval" / "audio_summary.json"),
                "visual_summary": str(run_dir / "eval" / "visual_summary.json"),
            }
        )
    OUT_JSON.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    headers = [
        "dataset",
        "joint_steps",
        "MAG",
        "ENV",
        "LRE",
        "DPAM",
        "RTE",
        "PSNR",
        "MSE",
        "L1",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_metric(row, key) for key in headers) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
