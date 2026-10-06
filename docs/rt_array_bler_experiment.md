# Sionna RT 基站阵列规模与 DMRS / beam-independent BLER

- **实验日期：** 2026-10-06
- **问题：** 在 Sionna RT 四用户上行 PUSCH 链路中，把 BS 阵列从 8×8 扩大、收窄接收波束，是否能改善 `dmrs` 信道估计 + `beam-independent` 检测器的 BLER？
- **结论：** 在本文所测固定 `ground_wall` RT 快照上，16×16 阵列相对 8×8 显著改善了空间隔离和端到端 BLER。8×8 在 20/25/30 dB 的每个 UE 都是 500/500 TB 错误；16×16 在同样的参考 SNR 上是 0/500。另一个保持阵元噪声方差不变的对照也得到 16×16 零错误。此结果支持“增大阵列有助于本配置”，但不等于一般场景中的 BLER 保证或统计衰落基准。

## 1. 实现背景与比较目标

RT 链路为 4 个 UE 各形成一个接收波束。代码根据 BS 与 UE 配置位置计算波束方向，以阵元坐标和载波波长生成单位范数复数权重，再得到等效信道

\[
H_{\mathrm{beam}} = W^{\mathsf H}H_{\mathrm{ant}}.
\]

该接收权重是几何指向的相位型权重，不是从接收 CSI 在线学习出的 MVDR/MMSE 波束；`phase_bits = 0` 表示这次试验没有相位量化。RT BLER 仿真使用合并后的四路波束信道，`beam-independent` 检测器逐路均衡，并把其他流视为干扰。实现与定义见 `src/nr_pusch/beamforming.py:13-66,79-113`、`src/nr_pusch/rt_channel.py:217-249,903-920`、`src/nr_pusch/receiver.py:164-201`。

本实验测的是完整 `dmrs` + `beam-independent` PUSCH 接收链路；不是单独的 DMRS CSI 误差测试，也不是对所有检测器的比较。

## 2. 配置与控制变量

### 2.1 RT 场景和阵列

基准来自 `configs/rt_beam_ground_wall.toml:1-53`：

| 参数 | 设定 |
|---|---|
| RT 场景 | 包内 `ground_wall`；`max_depth = 2`，`synthetic_array = true` |
| 载频、RT seed | 3.5 GHz、13 |
| 路径追踪预算 | 100,000 samples/source；最多 10,000 paths/source |
| BS | `[0, 0, 25]` m；朝向 `[0, 0, 0]` |
| UE | `ue0`–`ue3`，位置按 `configs/rt_beam_ground_wall.toml:23-45`；每 UE 发射功率缩放 0 dB |
| 阵元 | isotropic、V 极化；横/纵间距均为 0.5λ |
| 波束 | 方位/俯仰偏置均为 0°；`phase_bits = 0` |
| 噪声比 | `post_combiner_ratio = 0.001` |

**唯一的 RT 配置差异：** 8×8 基准改为 16×16；载频、阵元间距、BS/UE 位置、方向、功率、RT seed、路径预算和波束配置不变。阵元数由 64 增至 256；每轴端到端阵列孔径由 `7×0.5λ = 3.5λ` 增至 `15×0.5λ = 7.5λ`。因此主瓣理论上变窄，但这次没有直接测量 3 dB 波束宽度。

两份快照的 scene-bundle SHA-256 相同：`207839224ce0bb0602a7fb55ff50af7a8ef4da61659a64d4a13805633996a024`；均有 4 条有效路径/UE，且 `cp_sufficient = true`。8×8 与 16×16 的快照 `array_sha256` 分别为 `a46151c03187e73d46bd972cb6f0b503b8e697f85f557fc5626875bc78144a23` 和 `c3617890e5d1a0726dd58cb0787cfd11e0d9ef4c0b13d94d587af43c703dac44`。对应元数据在 `/tmp/nr-rt-array-sweep/8x8/channel.json`、`/tmp/nr-rt-array-sweep/16x16/channel.json`。

### 2.2 发射与 BLER 设置

发射配置为 `configs/pusch_4ue.toml:1-42`：50 RB、30 kHz 子载波间隔、DFT-s-OFDM、MCS table 1/index 20、4 UE；RT BLER 入口要求每 UE 单层、单发射天线（`src/nr_pusch/beam_simulation.py:76-78`）。两种阵列共同使用：

```toml
[bler]
snr_db = [20.0, 25.0, 30.0]
batch_size = 16
max_frames_per_snr = 500
target_block_errors = 10000
seed = 13
num_decoder_iterations = 20
channel_estimator = "dmrs"
channel_estimators = ["dmrs"]
detector = "beam-independent"
detectors = ["beam-independent"]
detector_damping = 0.25
device = "cuda:0"
channel_domain = "frequency"
l_min = -6
max_delay_spread_s = 3e-6
stop_at_zero_bler = false
```

`target_block_errors = 10000` 且 `stop_at_zero_bler = false`，因此每个 SNR 点都完整运行 500 帧；每帧 4 个 UE TB，即 500 TB/UE、2,000 TB/aggregate/SNR。`bler_ci95_low/high` 是每 UE 的双侧 95% Clopper–Pearson 区间；aggregate CI 是四 UE 98.75% 区间端点的平均，采用 Bonferroni 得到至少 95% simultaneous coverage，定义见 `src/nr_pusch/beam_simulation.py:370-374,594-613,834-844`。

两种阵列都使用 seed 13；但阵元噪声张量维度随阵元数从 64 变为 256，因此不应将这两次仿真视为逐样本配对的同一 AWGN realization。

## 3. BLER 结果

### 3.1 相同参考 SNR

| 阵列 | 参考 SNR | UE0 | UE1 | UE2 | UE3 | Aggregate |
|---|---:|---:|---:|---:|---:|---:|
| 8×8 | 20 dB | 500/500，BLER 1.0 | 500/500，1.0 | 500/500，1.0 | 500/500，1.0 | 2,000/2,000，1.0 |
| 8×8 | 25 dB | 500/500，BLER 1.0 | 500/500，1.0 | 500/500，1.0 | 500/500，1.0 | 2,000/2,000，1.0 |
| 8×8 | 30 dB | 500/500，BLER 1.0 | 500/500，1.0 | 500/500，1.0 | 500/500，1.0 | 2,000/2,000，1.0 |
| 16×16 | 20 dB | 0/500，BLER 0 | 0/500，0 | 0/500，0 | 0/500，0 | 0/2,000，0 |
| 16×16 | 25 dB | 0/500，BLER 0 | 0/500，0 | 0/500，0 | 0/500，0 | 0/2,000，0 |
| 16×16 | 30 dB | 0/500，BLER 0 | 0/500，0 | 0/500，0 | 0/500，0 | 0/2,000，0 |

每个 UE、每个 SNR 点均为 500 帧。8×8 每 UE 的 95% CI 为 `[0.992649, 1]`；16×16 每 UE 的 95% CI 为 `[0, 0.007351]`。aggregate simultaneous 95% CI 分别为 `[0.989901, 1]` 和 `[0, 0.010099]`。这不是“16×16 BLER 等于零”的证明；它表示本次每 UE 500 TB 中未观察到错误，95% 上界约为 0.735%。

8×8 aggregate BER 从 20/25/30 dB 的 `0.093810 / 0.078411 / 0.071116` 缓慢下降，但三个点仍然都是 100% TB BLER，符合严重残余干扰限制的表现。CSV 与 JSON manifest：

- `/tmp/nr-rt-array-sweep/8x8/results.csv`
- `/tmp/nr-rt-array-sweep/8x8/results.json`
- `/tmp/nr-rt-array-sweep/16x16/results.csv`
- `/tmp/nr-rt-array-sweep/16x16/results.json`

### 3.2 固定阵元噪声的对照

相同 `snr_db` 并不保持相同的绝对阵元噪声。RT BLER 代码先计算 UE0 参考数据 RE 信道功率 `q_ref`，再按

\[
\sigma_a^2 = \frac{q_{\mathrm{ref}}}{10^{\mathrm{SNR}_{\mathrm{ref}}/10}(G_{\mathrm{array}}+\rho)},
\quad \sigma_v^2=\rho\sigma_a^2
\]

确定阵元噪声和后级噪声；本实验两种阵列权重均为单位范数，`G_array = 1`，`ρ = 0.001`。公式见 `src/nr_pusch/beam_simulation.py:395-411,725-740`。

因此另外做了一个固定噪声点：比较 8×8 的 20 dB 与 16×16 的 `25.978613 dB`。两者 array noise variance 分别为 `1.3116329353e-9`、`1.3116330485e-9`，差约 `0.0000086%`；post-combiner variance 均约 `1.311633e-12`。16×16 在该点仍为每 UE 0/500 错误、aggregate 0/2,000。文件：`/tmp/nr-rt-array-sweep/16x16/same_element_noise.csv`、`/tmp/nr-rt-array-sweep/16x16/same_element_noise.json`。

此固定噪声点仅把 §2.2 的 `snr_db` 改为 `[25.978613]`；其他 BLER 参数不变。

## 4. 阵列与空间隔离指标

### 4.1 UE0 参考链路功率

由仿真 manifest 中 data-RE `q_ref` 得到：

| 阵列 | `q_ref` | 相对 8×8 |
|---|---:|---:|
| 8×8 | `1.3129445682e-7` | 0 dB |
| 16×16 | `5.2012499941e-7` | `3.9617×`，`+5.978613 dB` |

`q_ref` 按 `mean_data_RE(|H_beam[ue0,ue0]|²)` 计算；定义见 `src/nr_pusch/beam_simulation.py:127-134,395-402`。约 +6 dB 的增长与阵元数增加 4 倍时的理想相干接收增益相符。

### 4.2 非目标 UE 泄漏

对快照的 600 个 FFT bin 计算 `compute_beam_metrics`。该函数沿频率轴平均信道功率，并以每个目标波束的对角信道作分母，计算每个非目标 UE 的 ISR；定义见 `src/nr_pusch/beamforming.py:79-113`。四个 UE 功率均为 0 dB。

| 指标 | 8×8 | 16×16 |
|---|---:|---:|
| 最差非对角 beam/user ISR | `0.0661185`（−11.80 dB） | `0.00185517`（−27.32 dB） |
| 最大非目标 ISR 降幅 | — | 约 `15.52 dB`，约 35.6 倍 |
| `|WᴴW|` 最大非对角元素 | `0.220994` | `0.0429734` |
| 按 FFT bin 的最小奇异值/最大奇异值比 | `0.3840` | `0.8331` |

这些是快照信道诊断，不是额外的 BLER 试验；非对角 ISR 是全 600-bin 平均后取最坏 beam/user 对，不表示每个子载波、每条路径都恰好改善 15.52 dB。更小的非对角 `WᴴW` 也意味着阵元噪声经合并后跨波束相关性减弱；噪声协方差模型为 `src/nr_pusch/noise.py:71-82` 中的 `σ_a²WᴴW + σ_v²I`。

## 5. 结论与适用范围

1. **本场景支持使用更大孔径改善 DMRS + beam-independent。** 同参考 SNR 下，8×8 在三个测试点全部 TB 失败，而 16×16 在三个点均无错误；匹配阵元噪声的额外对照仍得到相同方向的结果。
2. **可见机制是空间选择性和阵列增益同时改善。** 参考数据 RE 信道功率增加约 5.98 dB，最差非目标 ISR 下降约 15.52 dB。由于 beam-independent 均衡器逐端口处理、将其他流作为干扰，更低的非目标泄漏直接有利于该模式。
3. **不能由这些数据外推一个精确 coding gain 或部署保证。** SNR 点上结果已经饱和：8×8 在 30 dB 仍 100% BLER，16×16 在 20 dB 已 0/500；需要在转折区加密 SNR 才能估算曲线位移。
4. **结论是固定场景条件 BLER，不是跨信道统计。** `simulate_rt_beam_bler` 对一个静态 RT realization 生成多帧，不逐帧重追踪或重采样传播信道，见 `src/nr_pusch/beam_simulation.py:57-67,435-438`。当前 snapshot 是 `ground_wall`、每 UE 4 条有效路径、CP 通过；置信区间只描述该固定快照上的帧/噪声 Monte Carlo，不覆盖用户位置、场景、遮挡或移动变化。
5. **未建模真实阵列硬件限制。** `phase_bits=0` 使用理想连续相位权重；本实验未覆盖相位校准误差、互耦、馈线/RF 链差异、阵元失效或机架尺寸/成本。扩阵上线前应以真实可实现的相位精度和多组场景/UE 位置复测，并与 ZF/LMMSE 联合检测对照。

## 6. 重现信息

本次所有新增配置与工件放在 `/tmp/nr-rt-array-sweep/`，未修改仓库配置或源码。8×8 输入为 `configs/rt_beam_ground_wall.toml`；16×16 输入是该配置副本，仅将 `receiver.num_rows`、`receiver.num_cols` 从 8 改为 16。BLER 配置见 §2.2；固定噪声配置与输出见 §3.2。快照由 `nr_pusch.cli.rt_channel` 生成，固定快照 BLER 由 `nr_pusch.cli.beam_bler` 运行，CLI 参数见 `src/nr_pusch/cli/rt_channel.py:22-74`、`src/nr_pusch/cli/beam_bler.py:23-54,78-100`。

示例调用（16×16 使用本次临时 RT 配置；若 `/tmp` 工件已清理，按上述唯一配置差异重建）：

```bash
PYTHONPATH=src /home/le-lei/workspace/test/.venv/bin/python -m nr_pusch.cli.rt_channel \
  --tx-config configs/pusch_4ue.toml \
  --rt-config /tmp/nr-rt-array-sweep/rt_beam_ground_wall_16x16.toml \
  --cache-dir /tmp/nr-rt-array-sweep/cache \
  --output /tmp/nr-rt-array-sweep/16x16/channel.npz

PYTHONPATH=src /home/le-lei/workspace/test/.venv/bin/python -m nr_pusch.cli.beam_bler \
  --tx-config configs/pusch_4ue.toml \
  --channel-snapshot /tmp/nr-rt-array-sweep/16x16/channel.npz \
  --simulation-config /tmp/nr-rt-array-sweep/bler_same_reference_snr.toml \
  --seed 13 --device cuda:0 \
  --output /tmp/nr-rt-array-sweep/16x16/results.csv
```

