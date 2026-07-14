# AVGaussianFusion v2 技术方案与实验汇报

## 1. 目标

我们希望把 FreeTimeGS++ 的动态视觉高斯场和 AudioGS 的空间音频建模能力融合起来，形成一个可同时重建视频和双耳音频的音视频高斯场。

核心目标：

- 保留 FreeTimeGS++ 在动态视觉重建上的稳定性。
- 利用视觉高斯提供的几何、运动和视角信息约束音频重建。
- 面向 held-out camera 重建目标视角 RGB 与目标视角双耳音频。
- 在不破坏视觉质量的前提下，提高音频幅度、包络和左右声道空间关系。

## 2. 当前 latest 方案：Route C + stereo regularization

当前推荐方案是 **Route C soft-coupled acoustic Gaussians + stereo regularization**。

整体思路是：

1. 先用 FreeTimeGS++ 训练动态视觉高斯场，得到稳定的视觉 carrier。
2. 冻结 FreeTimeGS++ 视觉场，不直接改动视觉高斯参数。
3. 从视觉高斯初始化一组独立的 acoustic Gaussians。
4. acoustic Gaussians 保留自己的声学属性和运动参数。
5. 通过软约束把 acoustic Gaussians 绑定到视觉 anchor，而不是强行共享同一套参数。
6. 用频域 transfer renderer 把 source view 音频变换为 target view 双耳音频。

这样做的关键点是：视觉负责稳定几何和动态场，音频负责学习独立的声学传播和双耳差异，二者通过 soft coupling 协同，而不是互相拖累。

## 3. 模型结构

### 3.1 Visual carrier：FreeTimeGS++

FreeTimeGS++ 提供动态视觉状态：

- 3D Gaussian 位置、透明度、运动等参数。
- 任意时间 `t` 下的动态视觉高斯状态。
- held-out camera 下的 RGB 渲染能力。

在当前 Route C 中，FreeTimeGS++ 是冻结的视觉底座。视觉 PSNR 直接继承 FreeTimeGS++，避免音频训练破坏视觉重建。

### 3.2 Acoustic Gaussian Field

在视觉高斯基础上初始化 acoustic Gaussians。每个 acoustic Gaussian 包含：

- `xyz`：声学高斯位置。
- `velocity`：声学运动。
- `audio_opacity`：声学活跃度。
- `mono_response`：整体频率 transfer。
- `diff_response`：左右声道差异基础项。
- `diff_directional_response`：方向相关的左右声道差异。
- `side_response`：mid-side 中 side 分量的保持或增强。
- `distance_decay`：距离衰减。
- `phase_delay`：可选相位延迟。

声学高斯不是视觉高斯的硬拷贝。它们可以独立学习声学属性，但通过 anchor/motion/activity loss 保持与视觉动态场的关系。

### 3.3 Frequency Transfer Renderer

音频渲染器工作在 STFT 频域。

输入：

- source camera 的双耳音频。
- target camera 的外参。
- 时间 `t` 下的 acoustic Gaussian 状态。

处理流程：

1. 对 source audio 做 STFT。
2. 分解为 mid / side：
   - `mid = 0.5 * (L + R)`
   - `side = 0.5 * (L - R)`
3. 根据 acoustic Gaussians、相机方向和距离估计频率 transfer。
4. 生成 target view 的 left / right transfer。
5. iSTFT 回到时域双耳音频。

当前 latest 的关键增强是 directional stereo transfer：

- 使用 target camera 坐标系下的方向 `direction = xyz_camera / distance`。
- `side = direction_x` 表示左右方向性。
- `diff_directional_response · direction` 建模与视角方向相关的双耳差异。
- 可选 `geometry_diff_head` 根据方向、前后、距离、source ILD、source side magnitude 和频率预测额外左右差异。

这使模型不只是学习一个固定的 L/R gain，而是能根据相机视角动态改变双耳空间关系。

## 4. 训练目标

当前训练以音频重建为主，视觉场冻结。

主要 loss：

- audio reconstruction：AudioGS-style mono/diff 重建损失。
- LRE / band LRE：约束左右能量比。
- TF-ILD：约束时频域 interaural level difference。
- phase-diff：约束左右声道相位差。
- mono-diff phase：约束 mid/side 相位关系。
- coherence：约束左右声道相关性。
- energy balance：约束整体能量。
- diff/side response regularization：约束频率响应不要过度抖动或放大。
- soft coupling：anchor、motion、activity、sparsity 约束。

训练 schedule：

- acoustic warmup：先只训练声学场，coupling 权重为 0。
- joint stage：逐步增加 soft coupling 权重。
- 当前 stereo reg 配置默认 `3000 + 1000` steps。
- 使用 `0.5s` center window，同时以 `70%` 概率混入 `3s` long window，提高与 AudioGS-style full-track 评测的一致性。

## 5. 技术难点与原理

### 5.1 难点一：视觉高斯和声学高斯不是同一种物理量

FreeTimeGS++ 的高斯主要服务于图像渲染，优化目标是颜色、透明度、几何和动态轨迹。AudioGS 的高斯服务于声音传播，优化目标是声源到听者的频率响应、能量变化和双耳差异。

如果直接把二者硬绑定，会遇到两个问题：

- 视觉上重要的高斯，不一定是声学上重要的反射或传播位置。
- 声学需要调整的参数，可能会破坏视觉几何和 RGB 质量。

当前 Route C 的原则是：**共享结构先验，不共享全部参数**。视觉高斯提供初始化和 anchor，声学高斯保留独立的音频属性和一定运动自由度。

### 5.2 难点二：音频不是单帧属性，而是短时间窗口信号

视频重建通常按时间 `t` 渲染一帧 RGB；音频重建需要处理一段 waveform。一个视觉时间点只对应一个瞬时状态，但声音窗口内包含多个 STFT time bins。

当前做法是：

- 用视觉帧时间 `t` 查询 acoustic Gaussian state。
- 以同一个 `t` 为中心裁剪 source / target audio window。
- 在短窗口内认为场景声学 transfer 相对稳定。
- STFT 的时间维只表示局部音频变化，不再重新查询不同视觉状态。

这是一种务实折中：既保证音视频同步，又避免每个 STFT bin 都查询动态高斯导致训练复杂度过高。

### 5.3 难点三：双耳音频的核心不是响度，而是左右关系

单声道或整体幅度拟合好，并不代表空间音频好。双耳音频至少要拟合：

- 左右声道能量差，即 ILD / LRE。
- 左右声道相位差。
- mid / side 结构。
- 不同频段上的空间差异。
- 不同视角下空间差异随几何方向变化。

因此 latest 方案不只优化 waveform 或 magnitude，而是显式加入 TF-ILD、band LRE、phase-diff、mono-diff phase、coherence 和 energy balance。

### 5.4 难点四：目标视角音频依赖相机几何

从 source view 音频生成 target view 音频，本质上是在估计一个视角相关的 transfer function。这个 transfer 不应是固定的，因为 target camera 改变后，左右方位、距离和遮挡关系都会变化。

当前 renderer 的核心原理是：

1. 把 acoustic Gaussian 位置变换到 target camera 坐标系。
2. 计算距离 `distance` 和方向 `direction`。
3. 用距离控制衰减。
4. 用 `direction_x` 表示左右方位。
5. 用 `diff_directional_response · direction` 建模方向相关的 L/R 差异。
6. 得到 left / right transfer，再作用到 source audio 的 STFT 上。

这使模型具备基本的视角相关空间音频建模能力。

### 5.5 难点五：短窗同步和长窗评测存在冲突

训练时使用短窗口有利于音视频同步，因为一个视觉状态更容易解释附近的音频。但 AudioGS-style 评测通常使用 3s non-overlap full-track，长窗口更关注连续音频质量。

如果只用短窗训练，可能 full-track 拼接不稳；如果只用长窗训练，一个视觉状态又很难解释长时间声音变化。

当前 latest 使用 mixed window training：

- `0.5s` center window 保持精确音视频对齐。
- `3s` long window 提高与 full-track 评测协议的一致性。
- stereo regularization 约束窗口内的左右关系，减少拼接后 L/R 漂移。

### 5.6 原理总结

当前方法可以理解为一个视觉引导的频域声学 transfer 模型：

- FreeTimeGS++ 负责学习动态场景的视觉几何和时间结构。
- Acoustic Gaussians 继承视觉结构先验，但独立学习声学传播属性。
- Soft coupling 让声学场不要偏离视觉场太远。
- Frequency transfer renderer 在 STFT 域中学习 source view 到 target view 的频率响应。
- Directional stereo transfer 让左右声道差异随目标视角变化。
- Stereo losses 直接约束双耳空间感，而不是只追求波形幅度相似。

简言之：**视觉提供“在哪里、怎么动”，音频学习“怎么传播、左右怎么变”。**

## 6. 方案演进

### 6.1 Route A：Frozen visual carrier + audio head

第一版方案较直接：

- FreeTimeGS++ 作为 frozen carrier。
- 在视觉高斯上训练 AudioGS-style audio head。
- 优点是实现简单，视觉质量稳定。
- 问题是音频建模能力有限，尤其左右声道空间关系不够好。

结果上，Route A 能跑通融合流程，但从自身协议看音频误差仍偏高，说明简单挂 audio head 不够。

### 6.2 Route B：Joint AV fine-tuning

第二版尝试让视觉和音频共享高斯，并做联合微调：

- 从 Route A / FreeTimeGS++ 初始化。
- 训练 audio head，同时微调部分共享 Gaussian 参数。
- 设计 warmup / no-warmup / spectral head / strict AudioGS audio 等 ablation。

观察：

- 视觉 PSNR 略有提升，因为视觉也参与优化。
- 音频有一定改善，但不稳定。
- 硬共享会让视觉几何和声学传播互相牵制。
- 左右声道空间指标仍然不理想。

因此 Route B 更适合作为 hard-sharing ablation，不是当前主方案。

### 6.3 Route C：Soft-coupled acoustic Gaussians

第三版改为软耦合：

- 视觉 FreeTimeGS++ 冻结。
- 额外训练 acoustic Gaussian field。
- acoustic points 从视觉状态初始化，但保留独立参数。
- 通过软约束维持与视觉场的结构关系。

优点：

- 视觉质量稳定，不被音频训练破坏。
- 音频场有足够自由度表达声学传播。
- 可以自然加入频域 transfer、距离衰减、方向性差异。

基础 Route C 已经显著降低部分场景的 LRE，但 stereo 仍不够稳。

### 6.4 Latest：Route C + stereo regularization

最新方案在 Route C 上补强双耳空间建模：

- 加入 directional diff response。
- 加入 side response。
- 加入 TF-ILD、band LRE、phase、coherence、energy balance 等 stereo loss。
- 加入 response L2 和 smooth regularization，防止左右差异被过度放大。
- 引入 mixed window training，兼顾短窗口同步和 3s AudioGS-style 评测。

这是当前最完整、最推荐的方案。

## 7. 实验设置

数据集：

- `scene1_opera`
- `scene7_playing_300`

统一设置：

- 训练相机：38 cams。
- 测试相机：`cam10`。
- 音频源：`near.wav`。
- 视觉 baseline：FreeTimeGS++。
- 音频 baseline：独立 AudioGS source-code eval。

主要指标：

- `MAG`：频谱幅度误差，越低越好。
- `ENV`：包络误差，越低越好。
- `LRE`：左右能量比误差，越低越好。
- `DPAM` / `RTE`：AudioGS 原始评测中的感知与时延相关指标，部分 full-track 协议不可用。
- `PSNR`：视觉重建质量，越高越好。

## 8. 实验结果

说明：不同评估协议的指标不直接横向比较。本节按协议拆分表格，同一表格内只比较同一协议；两个数据集放在同一张表中展示。

### 8.1 主表：AudioGS-style 3s non-overlap full-track

这是当前最主要的音频对比协议。它按 AudioGS-style 将整段音频切成 `3s` 非重叠窗口，渲染后拼接为 full-track，再计算 `MAG / ENV / LRE`。

| 数据集 | 方法 | MAG ↓ | ENV ↓ | LRE ↓ | PSNR ↑ |
|---|---|---:|---:|---:|---:|
| scene1_opera | AudioGS full-track | 0.1274 | 0.1014 | 0.9802 | - |
| scene1_opera | Route C | 0.1320 | 0.0943 | 0.2963 | 26.68 |
| scene1_opera | Route C + stereo reg | 0.1281 | 0.0972 | 0.0405 | 26.68 |
| scene7_playing_300 | AudioGS full-track | 0.0625 | 0.0328 | 0.1773 | - |
| scene7_playing_300 | Route C | 0.0669 | 0.0327 | 0.8018 | 21.03 |
| scene7_playing_300 | Route C + stereo reg | 0.0633 | 0.0316 | 0.4088 | 21.03 |

分析：

- `scene1_opera` 上，Route C + stereo reg 的 `LRE` 从 Route C 的 `0.2963` 降到 `0.0405`，降幅约 `86.3%`；相比 AudioGS full-track 的 `0.9802` 也明显更低。
- `scene1_opera` 上，Route C + stereo reg 的 `MAG` 与 AudioGS 基本持平，`ENV` 略优。
- `scene7_playing_300` 上，Route C + stereo reg 相比 Route C 同时改善 `MAG / ENV / LRE`，其中 `LRE` 降幅约 `49.0%`。
- `scene7_playing_300` 上，Route C + stereo reg 的 `MAG / ENV` 接近 AudioGS，但 `LRE` 仍高于 AudioGS，说明该场景双耳空间关系更难。
- 两个数据集上，Route C + stereo reg 都保持 FreeTimeGS++ 的视觉 PSNR，不牺牲视觉质量。

### 8.2 子表：Route C visual-center 0.5s overlap-add full-track

该协议以视觉帧时间为中心裁剪 `0.5s` 音频窗口，用对应时间的 acoustic Gaussian state 渲染，再通过 overlap-add 拼接为 full-track。它更强调音视频同步和短窗连续性，不与 `3s non-overlap` 结果直接比较。

| 数据集 | 方法 | MAG ↓ | ENV ↓ | LRE ↓ | PSNR ↑ |
|---|---|---:|---:|---:|---:|
| scene1_opera | Route C | 0.2045 | 0.1373 | 1.5340 | 26.68 |
| scene1_opera | Route C + stereo reg | 0.1979 | 0.1383 | 1.1904 | 26.68 |
| scene7_playing_300 | Route C | 0.0696 | 0.0346 | 0.7297 | 21.03 |
| scene7_playing_300 | Route C + stereo reg | 0.0660 | 0.0336 | 0.4318 | 21.03 |

分析：

- 在两个数据集上，stereo regularization 都降低了 overlap-add full-track 的 `LRE`。
- `scene1_opera` 的 overlap-add `LRE` 仍较高，说明短窗拼接后的左右声道连续性仍是难点。
- `scene7_playing_300` 上，Route C + stereo reg 在 `MAG / ENV / LRE` 都优于基础 Route C。
- 该协议更适合观察短窗同步和连续拼接问题，不作为与 AudioGS baseline 的主结论依据。

### 8.3 子表：Route A full-track source-to-heldout audio

该协议是 Route A 自身的 full-track source-to-heldout audio 评估，不与 AudioGS-style `3s non-overlap` 直接比较。

| 数据集 | 方法 | MAG ↓ | ENV ↓ | LRE ↓ | PSNR ↑ |
|---|---|---:|---:|---:|---:|
| scene1_opera | Route A frozen visual + audio head | 0.2214 | 0.1366 | 0.7420 | 26.68 |
| scene7_playing_300 | Route A frozen visual + audio head | 0.0902 | 0.0371 | 0.4590 | 21.03 |

分析：

- Route A 证明 FreeTimeGS++ carrier 上可以挂接音频重建头，流程可行。
- 但从该协议自身结果看，音频误差仍较高，尤其 `MAG` 和 `LRE` 不够理想。
- 这也是后续从简单 audio head 演进到独立 acoustic Gaussian field 的主要原因。

### 8.4 子表：Route B visual-frame window average

该协议在视觉帧对应的音频窗口上评估，并对窗口指标求平均，不是 full-track 拼接结果。因此它主要用于 Route B 内部 ablation。

| 数据集 | 方法 | MAG ↓ | ENV ↓ | LRE ↓ | PSNR ↑ |
|---|---|---:|---:|---:|---:|
| scene1_opera | Route B spectral head | 0.1860 | 0.1129 | 0.8015 | 26.88 |
| scene1_opera | Route B spectral head + strict AudioGS audio | 0.2067 | 0.1106 | 0.8361 | 26.88 |
| scene7_playing_300 | Route B spectral head | 0.0855 | 0.0324 | 0.3063 | 21.84 |
| scene7_playing_300 | Route B spectral head + strict AudioGS audio | 0.0764 | 0.0290 | 0.4438 | 21.84 |

分析：

- Route B 的视觉 PSNR 高于冻结视觉 baseline，说明 joint fine-tuning 对 RGB 有收益。
- 但音频指标不稳定，尤其 `LRE` 并没有稳定改善。
- 这支持当前判断：Route B 更适合作为 hard-sharing ablation，而不是最终主方案。

### 8.5 子表：FreeTimeGS++ visual-only

该协议只评估 held-out camera 的 RGB 重建，不包含音频指标。

| 数据集 | 方法 | PSNR ↑ |
|---|---|---:|
| scene1_opera | FreeTimeGS++ visual only | 26.68 |
| scene7_playing_300 | FreeTimeGS++ visual only | 21.03 |

分析：

- Route C 和 Route C + stereo reg 冻结 FreeTimeGS++，因此视觉 PSNR 与该 visual-only baseline 保持一致。
- Route B 因为做 joint fine-tuning，视觉 PSNR 略高，但也引入了视觉和音频互相牵制的问题。

## 9. 结论

当前 latest 方案可以概括为：

> 用 FreeTimeGS++ 提供稳定的动态视觉高斯 carrier；用独立 acoustic Gaussians 学习音频传播；通过 soft coupling 连接视觉和声学；通过 directional stereo transfer 和 stereo regularization 提升双耳空间音频重建。

当前结论：

- Route A 跑通了 FreeTimeGS++ + AudioGS 的基础融合，但音频能力不足。
- Route B 验证了 hard-sharing joint optimization，但视觉和音频互相牵制，稳定性不够。
- Route C 是更合理的结构：视觉冻结、声学独立、软耦合。
- Latest stereo regularization 显著改善 LRE，尤其 `scene1_opera` 上已经明显优于 AudioGS full-track。
- `scene7_playing_300` 上 MAG/ENV 已接近 AudioGS，但 LRE 仍有差距，说明双耳空间建模还需要加强。

## 10. 后续方向

建议优先推进：

1. 提升 0.5s visual-center overlap-add 协议下的连续性，减少窗口拼接造成的 L/R 不一致。
2. 加强时序平滑约束，让 stereo transfer 随时间连续变化。
3. 引入更明确的 source-target camera 几何关系，而不只使用 target camera 下的 acoustic point direction。
4. 对 `scene7_playing_300` 单独分析左右声道失败样例，定位 LRE 偏高原因。
5. 增加 full-track 可用的感知指标，减少只看 MAG/ENV/LRE 的片面性。

## 11. 关键代码入口

- Route C 训练：`avfusion/train/train_soft_av_gaussians.py`
- acoustic Gaussian field：`avfusion/soft/acoustic_field.py`
- frequency transfer renderer：`avfusion/soft/frequency_transfer_renderer.py`
- audio / stereo losses：`avfusion/audio/losses.py`
- Route C full-track eval：`avfusion/eval/eval_soft_audio_fulltrack.py`
- fair comparison：`avfusion/eval/fair_baseline_comparison.py`
- latest stereo reg config：`configs/scene1_opera_c_soft_av_gaussians_stereo_reg.yaml`
- latest stereo reg config：`configs/scene7_playing_300_c_soft_av_gaussians_stereo_reg.yaml`
