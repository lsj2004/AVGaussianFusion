# Route B Joint AV Gaussian Design

## Goal

Build a second route for AVGaussianFusion that performs end-to-end audio-visual Gaussian fine-tuning while preserving the existing Route A frozen-carrier baseline.

Route B starts from the trained FreeTimeGS++ visual Gaussian checkpoint, reuses the original FTGS++ differentiable renderer for RGB supervision, and attaches a trainable audio head to the same Gaussian set. Visual and audio losses then update shared Gaussian parameters plus audio-specific parameters in one training loop.

## Non-Goals

- Do not replace or rewrite Route A.
- Do not remove `train_frozen_carrier_audio.py`, the current lightweight audio renderer, or existing Route A scripts.
- Do not implement full AudioGS time-frequency-bin optimization in the first Route B version.
- Do not use DPAM as a training loss in the first version; keep it for evaluation.
- Do not retrain FTGS++ from scratch inside the Route B trainer. Route B consumes an existing FTGS++ checkpoint.

## Current Baseline To Preserve

Route A currently does:

1. Train FTGS++ visual Gaussians.
2. Export a frozen visual carrier.
3. Select top-k high-opacity visual Gaussians as an acoustic carrier.
4. Freeze visual geometry.
5. Train only lightweight audio parameters: `mono_gain` and `diff_gain`.
6. Evaluate RGB metrics and AudioGS-style audio metrics.

Route B must live beside this and use separate modules, configs, scripts, checkpoints, and output directories.

## Architecture

Route B introduces a joint model:

```text
FTGS++ gaussians.pt
        |
        v
JointAVGaussianModel
        |
        +-- FTGS++ Gaussians.forward(t, w2c, intrinsic, shape) -> RGB image -> visual losses
        |
        +-- AudioGaussianHead(shared Gaussian state) -> stereo audio -> audio losses
```

The visual branch must call the original FTGS++ renderer:

```python
ftgspp.models.gaussians.Gaussians.forward(
    t,
    w2c,
    intrinsic,
    shape,
    sh_degree=sh_degree,
)
```

That function already calls `gsplat.rasterization`, so visual losses can backpropagate into the shared Gaussian parameters.

The audio branch queries the same dynamic Gaussian state at the training time `t`, selects or weights points by opacity, and predicts stereo audio from source audio and trainable acoustic attributes.

## New Files

```text
avfusion/joint/__init__.py
avfusion/joint/ftgspp_bridge.py
avfusion/joint/model.py
avfusion/joint/audio_head.py
avfusion/joint/losses.py
avfusion/train/train_joint_av_gaussians.py
configs/scene1_opera_b_joint_av.yaml
scripts/train_joint_scene1_opera.sh
scripts/eval_joint_scene1_opera.sh
tests/test_joint_ftgspp_bridge.py
tests/test_joint_audio_head.py
tests/test_train_joint_av_gaussians.py
```

## Parameter Ownership

### Shared Gaussian Parameters

Loaded from FTGS++:

```text
means
scales
quats
opacities
sh_0
sh_n
times
durations
velocity_model
marginal_gates
```

The first Route B implementation uses conservative fine-tuning:

```text
trainable:
  means
  opacities
  explicit velocities when available
  audio head parameters

frozen or very low learning rate:
  scales
  quats
  sh_0
  sh_n
  times
  durations
  marginal_gates
```

This keeps the visual reconstruction from collapsing while still allowing audio to influence geometry.

### Audio-Specific Parameters

The initial joint audio head extends Route A:

```text
audio_opacity      (N, 1)
mono_gain          (N, 1)
diff_gain          (N, 1)
delay_offset       (N, 1)
attenuation_logit  (N, 1)
```

`N` is the number of selected or retained shared Gaussians. The implementation may use `top_k` selection from the shared Gaussian opacity for memory control, but the selected points still refer back to the shared FTGS++ Gaussian tensors so gradients can reach shared geometry.

## Data Flow

Each training step samples:

```text
visual sample:
  frame, camera, rgb, w2c, intrinsic, time

audio sample:
  source_audio, target_audio, camera, time/crop metadata
```

For the first version, visual and audio samples may be paired by the same scene and approximate time. If exact audio-video frame alignment is unavailable, the trainer should use the manifest crop time and document the alignment assumption in the checkpoint metadata.

## Training Stages

### Stage 0: Initialization

Inputs:

```text
runs/scene1_opera_a/ftgspp/scene1_opera/00/gaussians.pt
runs/scene1_opera_a/scene_manifest.json
configs/ftgspp_scene1_opera/scene1_opera.toml
```

Route B loads the FTGS++ checkpoint in an environment where FTGS++ and `gsplat` are available.

### Stage 1: Audio Warmup

Only the audio head is trainable. Shared FTGS++ Gaussian parameters are frozen. This establishes a Route-A-compatible starting point and makes Route B comparable against the existing frozen-carrier baseline.

Loss:

```text
L_audio_warmup = lambda_mag * L_MAG_train
               + lambda_lre * L_LRE_train
               + lambda_wave_debug * L_waveform_optional
```

### Stage 2: Joint Fine-Tuning

Open a controlled subset of shared Gaussian parameters and train with visual and audio losses together.

Loss:

```text
L_total = lambda_rgb_l1  * L_RGB_L1
        + lambda_rgb_ssim * L_RGB_SSIM
        + lambda_mag     * L_MAG_train
        + lambda_env     * L_ENV_train
        + lambda_lre     * L_LRE_train
        + lambda_xyz_reg * ||means - means_init||
        + lambda_op_reg  * ||sigmoid(opacity) - sigmoid(opacity_init)||
        + lambda_vel_reg * ||velocity - velocity_init||
```

DPAM remains evaluation-only.

## Renderer Details

### Visual Renderer

The bridge should be thin:

```python
class FTGSRendererBridge:
    def load(path) -> Gaussians
    def render_rgb(gs, batch, sh_degree) -> torch.Tensor
```

It should not copy FTGS++ rasterization code into AVGaussianFusion. It should import and call FTGS++ directly so future FTGS++ renderer behavior stays consistent.

### Audio Renderer

The first joint audio renderer remains lightweight but geometry-aware:

1. Query dynamic positions at time `t`.
2. Compute per-point opacity weights.
3. Optionally include distance attenuation from listener/camera position.
4. Apply learned delay offsets with differentiable fractional delay or a first-version integer/linear interpolation approximation.
5. Aggregate mono and left-right differential components into stereo output.

The audio renderer must expose gradients to at least:

```text
audio head parameters
selected shared means
selected shared opacities
selected shared velocities
```

## Evaluation

Route B uses the existing visual and audio metrics:

Visual:

```text
PSNR
SSIM
LPIPS
DSSIM
EPE if flow evaluation is enabled
```

Audio:

```text
MAG
ENV
LRE
DPAM
RTE optional, unavailable until RT60 estimator integration is configured
```

The final report must compare:

```text
Route A frozen carrier baseline
Route B warmup checkpoint
Route B joint fine-tuned checkpoint
```

## Checkpoints

Route B checkpoint format should include:

```python
{
    "route": "B_joint_av",
    "ftgspp_checkpoint": "/absolute/path/to/source/gaussians.pt",
    "manifest_path": "/absolute/path/to/scene_manifest.json",
    "shared_gaussians": gs,
    "audio_head": audio_head.state_dict(),
    "config": resolved_config,
    "stage": "warmup" | "joint",
    "loss_history": list_of_step_metrics,
}
```

Saving `shared_gaussians` keeps visual rendering reproducible with FTGS++ tools. The checkpoint should also store enough config to rerun evaluation without guessing protocol settings.

## Error Handling

- If FTGS++ cannot be imported, the joint trainer must fail with a message naming the required FTGS++ root and environment.
- If `gsplat` is missing, visual joint training must fail early.
- If the FTGS++ checkpoint contains a neural `VelocityField` that cannot import `tinycudann`, the trainer must either run in the FTGS++ environment or fail with a clear message. The CPU unpickle stub used for carrier export is not enough for joint training because visual rendering needs the real model.
- If DPAM cannot run, evaluation should still write MAG/ENV/LRE and mark DPAM unavailable.

## Tests

Unit tests:

- Load a fake FTGS-style Gaussian object through the bridge API.
- Verify joint model exposes separate parameter groups for shared visual parameters and audio head parameters.
- Verify audio head output shape is stereo `(2, samples)`.
- Verify geometry regularization is zero at initialization and positive after perturbation.
- Verify Route A tests still pass unchanged.

Integration smoke tests:

- Build a tiny synthetic joint model with a fake renderer and confirm one training step updates audio head parameters.
- In an FTGS++ environment, run `train_joint_av_gaussians.py --help`.
- On real `scene1_opera`, run a short warmup plus short joint fine-tune and write metrics to `runs/scene1_opera_b_joint_av/`.

## Open Constraints

- The first implementation will be environment-sensitive because it needs FTGS++ plus `gsplat`.
- Joint training can be memory-heavy; default config should use small batch sizes and top-k audio points first.
- Exact audio-video temporal alignment remains approximate for the current sampled data unless a stronger sync source is added.

## Success Criteria

The Route B implementation is successful when:

1. Route A commands and tests still work.
2. Route B can load the existing FTGS++ checkpoint.
3. Route B can render RGB through FTGS++ during training.
4. Route B can train audio head parameters in warmup mode.
5. Route B can run at least a short joint fine-tune where RGB and audio losses both contribute gradients.
6. Route B writes separate visual/audio metrics under `runs/scene1_opera_b_joint_av/`.
7. Results can be compared against Route A without overwriting Route A outputs.
