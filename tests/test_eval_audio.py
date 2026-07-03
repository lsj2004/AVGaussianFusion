import json

import torch

from avfusion.eval.eval_audio import write_eval_summary


def test_write_eval_summary_records_l1_waveform(tmp_path):
    output_path = tmp_path / "summary.json"
    pred = torch.zeros(4)
    target = torch.ones(4)

    summary = write_eval_summary(output_path, camera="cam01", pred=pred, target=target)

    assert summary == {"camera": "cam01", "l1_waveform": 1.0}
    assert json.loads(output_path.read_text()) == summary
