import json

import pytest
import torch

from avfusion.eval.eval_audio import evaluate_audio_checkpoint, main, write_eval_summary
from tests.test_audio_video_dataset import _write_manifest
from tests.test_train_frozen_carrier_audio import _write_carrier


def test_write_eval_summary_records_l1_waveform(tmp_path):
    output_path = tmp_path / "nested" / "summary.json"
    pred = torch.zeros(4)
    target = torch.ones(4)

    summary = write_eval_summary(output_path, camera="cam01", pred=pred, target=target)

    assert summary == {"camera": "cam01", "l1_waveform": 1.0}
    assert json.loads(output_path.read_text()) == summary


def test_write_eval_summary_rejects_shape_mismatch(tmp_path):
    with pytest.raises(ValueError, match="same shape"):
        write_eval_summary(
            tmp_path / "summary.json",
            camera="cam01",
            pred=torch.zeros(4),
            target=torch.zeros(2),
        )


def test_write_eval_summary_rejects_non_float_and_empty_tensors(tmp_path):
    with pytest.raises(ValueError, match="floating-point"):
        write_eval_summary(
            tmp_path / "summary.json",
            camera="cam01",
            pred=torch.zeros(4, dtype=torch.long),
            target=torch.zeros(4, dtype=torch.long),
        )

    with pytest.raises(ValueError, match="non-empty"):
        write_eval_summary(
            tmp_path / "summary.json",
            camera="cam01",
            pred=torch.zeros(0),
            target=torch.zeros(0),
        )


def test_evaluate_audio_checkpoint_writes_heldout_metrics(tmp_path):
    manifest = _write_manifest(tmp_path)
    carrier_path = tmp_path / "carrier.pt"
    checkpoint_path = tmp_path / "stage2_audio.pt"
    output_dir = tmp_path / "eval"
    _write_carrier(carrier_path)
    torch.save(
        {
            "carrier_path": str(carrier_path),
            "top_k": 2,
            "params": {
                "mono_gain": torch.zeros(2, 1),
                "diff_gain": torch.zeros(2, 1),
            },
            "loss_history": [0.1],
        },
        checkpoint_path,
    )

    summary = evaluate_audio_checkpoint(manifest, checkpoint_path, output_dir)

    assert summary["camera"] == "cam10"
    assert "l1_waveform" in summary
    assert "stft_magnitude" in summary
    assert (output_dir / "audio_summary.json").exists()


def test_eval_main_writes_summary(tmp_path):
    manifest = _write_manifest(tmp_path)
    carrier_path = tmp_path / "carrier.pt"
    checkpoint_path = tmp_path / "stage2_audio.pt"
    output_dir = tmp_path / "eval"
    _write_carrier(carrier_path)
    torch.save(
        {
            "carrier_path": str(carrier_path),
            "top_k": 2,
            "params": {
                "mono_gain": torch.zeros(2, 1),
                "diff_gain": torch.zeros(2, 1),
            },
            "loss_history": [0.1],
        },
        checkpoint_path,
    )

    main(
        [
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "audio_summary.json").exists()
