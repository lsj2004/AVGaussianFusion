# AVGaussianFusion

Route A prototype for `scene1_opera`: train FreeTimeGS++ as the dynamic visual Gaussian carrier, freeze it, then optimize AudioGS-style acoustic Gaussian attributes on top of that carrier.

## First Route A Commands

```bash
scripts/run_full_scene1_opera.sh
```

Or run each stage manually:

```bash
scripts/prepare_scene1_opera.sh
scripts/run_stage1_flow.sh
scripts/run_stage1_ftgspp.sh
scripts/export_stage1_carrier.sh /mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/ftgspp/scene1_opera/00/gaussians.pt
scripts/train_stage2_audio.sh
scripts/eval_scene1_opera.sh
```

The first implementation defaults are `cam10` held out, `near.wav` source audio,
3-second synchronized crops, and phase modeling disabled for the initial smoke
test.

Config:

```bash
configs/scene1_opera_a_frozen_carrier.yaml
configs/ftgspp_scene1_opera/scene1_opera.toml
```
