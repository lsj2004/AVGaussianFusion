# Frozen Visual Carrier + Optimizable AudioGS Renderer Design

Date: 2026-07-02

## Goal

Build the first research prototype for an audio-visual Gaussian field on `scene1_opera`.
The long-term goal is one Gaussian representation that supports both visual reconstruction/new-view synthesis and audio reconstruction/novel-view acoustic synthesis.

This first version intentionally chooses the conservative Route A:

- Train FreeTimeGS++ normally to obtain a dynamic visual Gaussian field.
- Freeze the visual Gaussian geometry and dynamics.
- Initialize an AudioGS-style differentiable audio renderer from that frozen visual carrier.
- Optimize only acoustic Gaussian attributes and audio rendering parameters.

This tests whether FreeTimeGS++ geometry is a useful spatial carrier for AudioGS-style acoustic rendering without immediately taking on full end-to-end joint optimization.

## Baselines And Inputs

Baseline repositories:

- Visual baseline: `/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus`
- Audio baseline: `/mnt/sda/lisujing/Dataset/audioGS-replay`

Initial scene data:

- Visual DyNeRF-style data: `/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera`
- AudioGS-ready audio: `/mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera`

Observed scene facts:

- 39 cameras: `cam00` through `cam38`
- Visual input: `poses_bounds.npy`, `cameras.npz`, and 39 `camXX.mp4` videos
- Visual timing: 30 fps, 150 frames
- Audio input: aligned 16 kHz stereo `camXX.wav` files plus `near.wav`
- Audio duration is about 5.184 seconds, which is close enough to the 150-frame / 30 fps visual clip for a first synchronized manifest.

## Correct Interpretation Of AudioGS

AudioGS should not be treated as a conventional neural network head.

For this project, "AudioGS-style renderer" means a set of optimizable acoustic Gaussian parameters plus a differentiable rendering process. The trainable state may include mono/diff spherical-harmonic parameters, attenuation parameters, phase or ITD residuals, and per-point acoustic weights. Training optimizes these parameters through audio reconstruction losses.

The Route A architecture is therefore:

```text
frozen FreeTimeGS++ visual/dynamic Gaussian carrier
  + optimizable AudioGS-style acoustic Gaussian attributes
  -> differentiable acoustic renderer
  -> binaural audio prediction
```

It is not:

```text
visual Gaussian features -> ordinary neural head -> audio
```

## Method

### Stage 0: Unified Scene Manifest

Create one scene manifest that aligns visual frames, camera poses, and audio windows. The manifest records:

- Scene id: `scene1_opera`
- Camera names and numeric ids
- Source visual root and source audio root
- FPS and visual frame count
- Audio sample rate and channel count
- Per-camera video and audio paths
- Per-frame time in seconds
- Audio window mapping for each frame or training segment
- Train, validation, and test camera splits

The first split should use one held-out camera for both visual and audio evaluation. The default first implementation uses `cam10` as the held-out camera and stores this choice in config.

### Stage 1: Visual Carrier Training

Run FreeTimeGS++ on the DyNeRF-style visual data. This stage owns visual reconstruction and visual new-view synthesis.

The output needed by later stages is a dynamic Gaussian carrier with:

- Gaussian means
- Scales
- Rotations
- Opacities
- Time centers and durations
- Velocity model or explicit velocities
- Scene normalization metadata
- Camera intrinsics and extrinsics

This stage may also render held-out RGB views to verify that the visual carrier is usable before acoustic optimization begins.

### Stage 1.5: Carrier Export

Export the trained FreeTimeGS++ state into a stable project-local carrier format. The carrier should preserve dynamic behavior, especially `xyz(t)` and temporal opacity, because the scene is a short dynamic performance rather than a static capture.

The exported carrier is read-only for Route A training. All tensors that represent visual geometry or visual dynamics must have `requires_grad=False` in Stage 2.

### Stage 2: Acoustic Gaussian Initialization And Optimization

Initialize an AudioGS-style acoustic renderer from the frozen visual carrier.

The first version should not require a strict one-to-one match between FreeTimeGS++ visual Gaussians and AudioGS time-frequency bins. Instead, an adapter maps visual Gaussians into acoustic carrier points. The adapter may:

- Select a subset of visual Gaussians by opacity, temporal support, or visibility
- Sample acoustic carrier points from visual Gaussian positions over time
- Resample or compress carrier points when the visual Gaussian count differs from the AudioGS acoustic grid
- Keep dynamic position and temporal gate information frozen while allowing acoustic attributes to optimize

Trainable acoustic state may include:

- Mono and diff SH coefficients
- Distance attenuation parameters
- Optional per-point attenuation exponents
- Optional geometry-guided phase residuals
- Optional left/right energy ratio parameters

Stage 2 loss is audio-only in Route A. RGB losses do not update the visual carrier during acoustic optimization.

### Stage 3: Evaluation

Evaluate the same held-out camera split for the audio and visual branches.

Required artifacts:

- Predicted held-out binaural audio WAVs
- Ground-truth aligned WAV references
- Audio metric summary
- Visual held-out render summary, if the Stage 1 visual output is available
- Run metadata containing config, split, source roots, checkpoint paths, and command lines

Initial audio metrics should include STFT-based reconstruction metrics that are already available locally. ENV, DPAM, MAG, LRE, or phase metrics can be added when the local environment supports them reliably.

## Project Layout

The new project should live at:

```text
/mnt/sda/lisujing/Dataset/AVGaussianFusion
```

Planned layout:

```text
AVGaussianFusion/
  README.md
  configs/
    scene1_opera_a_frozen_carrier.yaml
  avfusion/
    data/
      build_scene_manifest.py
      audio_video_dataset.py
    visual/
      export_ftgspp_carrier.py
      carrier.py
    adapters/
      visual_to_acoustic.py
    audio/
      acoustic_gaussians.py
      renderer.py
      losses.py
    train/
      train_frozen_carrier_audio.py
    eval/
      eval_audio.py
      eval_visual.py
  scripts/
    prepare_scene1_opera.sh
    run_stage1_ftgspp.sh
    export_stage1_carrier.sh
    train_stage2_audio.sh
    eval_scene1_opera.sh
  docs/
    method_a_frozen_visual_carrier.md
    superpowers/specs/
```

## Module Responsibilities

`avfusion/data/build_scene_manifest.py` creates the unified manifest from the visual and audio roots.

`avfusion/data/audio_video_dataset.py` loads camera-specific audio windows, camera poses, and optional visual timing information for Stage 2.

`avfusion/visual/export_ftgspp_carrier.py` reads a FreeTimeGS++ checkpoint and writes the project-local frozen carrier.

`avfusion/visual/carrier.py` defines a small interface for querying frozen carrier state, especially `xyz(t)`, temporal opacity, and static Gaussian attributes.

`avfusion/adapters/visual_to_acoustic.py` converts frozen visual Gaussian carrier data into the acoustic carrier layout used by the AudioGS-style renderer.

`avfusion/audio/acoustic_gaussians.py` defines trainable acoustic Gaussian attributes.

`avfusion/audio/renderer.py` implements or wraps the differentiable AudioGS-style rendering path.

`avfusion/audio/losses.py` contains STFT, mono/diff, L/R, LRE, and optional phase losses.

`avfusion/train/train_frozen_carrier_audio.py` trains acoustic parameters while keeping the visual carrier frozen.

`avfusion/eval/eval_audio.py` writes predicted audio and metric summaries.

`avfusion/eval/eval_visual.py` records or delegates visual evaluation from FreeTimeGS++.

## Error Handling And Reproducibility

The first implementation should fail early when:

- Expected camera names differ between visual and audio roots
- FPS, frame count, or audio duration cannot support the configured segment mapping
- A held-out camera is missing from either modality
- The FreeTimeGS++ carrier checkpoint cannot be loaded
- Any visual carrier parameter is accidentally trainable during Stage 2

Every run should save:

- Fully resolved config
- Manifest copy
- Git status or source snapshot metadata where possible
- Train/eval split
- Commands used for Stage 1, export, Stage 2, and eval

## Testing Strategy

Initial tests should be small and focused:

- Manifest builder test for `scene1_opera`
- Camera-name matching test between video and audio roots
- Time mapping test from 150 frames at 30 fps to 16 kHz audio windows
- Carrier freeze test proving visual carrier tensors do not require gradients
- Adapter shape test for visual-to-acoustic carrier conversion
- One CPU-safe loss test for STFT loss shapes

GPU integration tests should be script-driven rather than unit-test driven:

- Run a short FreeTimeGS++ visual stage or reuse an existing checkpoint
- Export a carrier
- Train Stage 2 for a tiny number of iterations
- Render one held-out audio prediction

## Route C And B Migration Path

Route A is designed to grow into later versions.

Route C can be added by letting the acoustic Gaussian field become separate from the visual field while adding coupling losses to keep visual and acoustic geometry aligned.

Route B can be added after Route A is stable by unfreezing selected visual carrier parameters and allowing audio losses and RGB losses to jointly optimize shared Gaussian parameters.

The implementation should keep these paths possible by separating:

- Frozen carrier interface
- Visual-to-acoustic adapter
- Trainable acoustic attributes
- Differentiable acoustic renderer
- Optimization policy

## First Implementation Defaults

The first implementation should use these defaults unless the user changes them before implementation begins:

- Held-out camera: `cam10`
- Source/reference audio for Stage 2: `near.wav`
- Acoustic carrier initialization: sample a configurable top-k subset from visible/high-opacity visual Gaussians, with the first value chosen in the implementation plan after inspecting the exported FreeTimeGS++ Gaussian count
- Phase modeling: disabled for the first smoke test, then enabled after magnitude learning is verified
- Audio windowing: train on AudioGS-compatible 3-second synchronized crops first; add full-clip evaluation after the renderer produces valid held-out audio
