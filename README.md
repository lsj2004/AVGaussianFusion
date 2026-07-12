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
For strict AV eval, `MAG`/`ENV`/`LRE` are averaged over aligned audio windows.
`DPAM` is intentionally off by default; enable sampled-window DPAM with
`python -m avfusion.eval.eval_joint_audio ... --include-dpam --dpam-max-windows 16`.

Route B config:

```bash
configs/scene1_opera_b_joint_av.yaml
configs/scene1_opera_b_joint_av_no_warmup.yaml
```

## Route C Soft-Coupled AV Gaussians

Route C is the recommended AVGaussianFusion v2 path. It keeps the FreeTimeGS++
visual field as a base-preserving reference and trains an independent dynamic
acoustic Gaussian transfer field. Each acoustic Gaussian carries geometry plus
frequency-transfer attributes, and the renderer predicts `H(f; τ, listener)` for
short-window binaural reconstruction. Acoustic points are initialized from the
visual state, but remain trainable and are coupled to visual anchors with soft
anchor/motion/activity losses instead of hard parameter sharing.

Time alignment uses a global scene time `τ` from the visual frame. Source and
target audio are cropped around the same `τ`, while STFT time bins remain local
to that short audio window. The default Route C configs use 0.5-second centered
audio windows to avoid forcing one visual state to explain long audio segments.

```bash
scripts/train_soft_av_scene1_opera.sh
scripts/train_soft_av_scene7_playing_300.sh
scripts/eval_soft_av_scene1_opera.sh
scripts/eval_soft_av_scene7_playing_300.sh
```

Route C config:

```bash
configs/scene1_opera_c_soft_av_gaussians.yaml
configs/scene7_playing_300_c_soft_av_gaussians.yaml
```

The default schedule preserves the two bases: an acoustic warmup stage runs with
zero coupling weight, then the joint stage ramps soft coupling from 0 to 1 while
the visual FTGS++ field remains frozen. STFT is used only as the audio
analysis/synthesis domain: the learned transfer curve `H(f; τ, listener)` is
broadcast over local STFT bins inside the short crop. Route B remains as the
hard-sharing ablation.
