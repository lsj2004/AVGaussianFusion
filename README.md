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

## Route B Joint AV Fine-Tuning

Route A frozen-carrier training remains the baseline and keeps writing only under
`runs/scene1_opera_a`. Route B initializes shared Gaussians from the Route A
FTGS++ checkpoint, then jointly fine-tunes selected shared Gaussian
geometry/opacity/velocity parameters with geometry regularization. Joint
fine-tuning aligns audio to the visual batch by cropping a short centered
waveform window around each visual frame time, then renders audio and RGB from
the same Gaussian state `t`.

The default Route B keeps the staged schedule: audio-head warmup, then AV joint
fine-tuning. Outputs are written to `runs/scene1_opera_b_joint_av`.

```bash
scripts/train_joint_scene1_opera.sh
scripts/eval_joint_scene1_opera.sh
```

The no-warmup Route B skips audio-only warmup and starts strict AV joint
fine-tuning from step 0 after visual initialization. Outputs are written to
`runs/scene1_opera_b_joint_av_no_warmup`.

```bash
scripts/train_joint_no_warmup_scene1_opera.sh
scripts/eval_joint_no_warmup_scene1_opera.sh
```

`scripts/eval_joint_scene1_opera.sh` reports AudioGS-style heldout audio metrics
for the Route B checkpoint, including `MAG`, `ENV`, `LRE`, `RTE`, and optional
`DPAM`, and writes heldout visual reconstruction metrics including `PSNR`.

Route B config:

```bash
configs/scene1_opera_b_joint_av.yaml
configs/scene1_opera_b_joint_av_no_warmup.yaml
```
