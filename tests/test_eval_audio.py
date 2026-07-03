import json

import pytest
import torch

from avfusion.eval.eval_audio import write_eval_summary


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
