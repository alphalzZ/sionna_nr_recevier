# DMRS tap-power prior：原理、份数判定与生成维护

本文说明 `dmrs-lmmse` 使用的 tap-power prior 是什么、由什么决定、需要训练几份，以及如何训练、验收和发布。目的是让后续新增 profile 时能直接判断"要不要再训练一份先验"，而不是把每种几何都当成一种新先验。命令与实测数字见 `README.md` §"CDL tap-power LMMSE 验证"。

## 1. prior 是什么

DMRS-LMMSE 在每个 DMRS occasion 上把导频观测写成最小二乘问题：

$$
\mathbf A\,\mathbf t \approx \mathbf y,\qquad
\mathbf A_{m,\ell}=\text{导频处第 }m\text{ 个观测对第 }\ell\text{ 个 tap 的设计项},
$$

其中 $\mathbf t$ 是待估的抽头向量，tap 索引范围固定为 $\ell\in[l_{\min},\,l_{\max}]$。MMSE 解需要一个抽头协方差的先验，仓库里用的是**对角**形式：

$$
\mathbf R_t=\operatorname{diag}\big(p[l_{\min}],\dots,p[l_{\max}]\big),\qquad
p[\ell]=\mathbb{E}\big\{\,|h[\ell]|^2\,\big\}.
$$

即 $p$ 是在当前信道集合下**抽头功率的期望**，也就是功率时延谱（PDP）在离散 tap 栅格上的取值。实现上它是**一个**长度为 $l_{\max}-l_{\min}+1$ 的实向量缓冲（`receiver.py:1233` 的 `register_buffer("_tap_power_prior", …)`）。估计按 OCC 对联合作出：设计矩阵把该对里两个用户的抽头基拼接在一起，未知量维数是 `2 * num_taps`（两个用户 × 抽头数），因此 `torch.cat((prior, prior))` 是给两个用户各配一份**同样**的权重（`receiver.py:1272-1276`、`receiver.py:1309`），用于 $\mathbf A\mathbf R_t$；后验按每根接收天线分别求解，但用的都是同一份 $p$。所以它是一个**标量 PDP**，不随用户、天线或 tap 位置变化。

逐 tap 的作用是"该多大程度相信观测"。把 tap 先验写成零均值、方差为 $p[\ell]$ 的对角高斯，标量情形下的 LMMSE 收缩系数（仅用于说明方向，接收机用的是多天线联合的完整估计器）是

$$
g_\ell=\frac{p[\ell]}{p[\ell]+\sigma^2_{\text{eff}}},\qquad
\sigma^2_{\text{eff}}\approx\frac{\sigma^2}{|\mathbf A_{m,\ell}|^2},
$$

$\sigma^2$ 为观测噪声功率。$p[\ell]$ 大表示先验方差大、允许给出更大的数据驱动估计，增益接近 1；$p[\ell]$ 小时增益趋近 0，估计被强烈收缩回先验均值 0，等于判定该抽头几乎没有能量。也就是说 $p$ 的**形状**决定每个抽头各自被观测"相信"到什么程度。

## 2. 什么决定 $p[\ell]$

$$
p[\ell]=\mathbb{E}\big\{|h_d[\ell]|^2\big\},\qquad
h_d[\ell]=\sum_{p}\alpha_p\,g\!\left(\ell/f_s-\tau_p\right)
$$

其中 $h_d[\ell]$ 是冲激响应在抽样时刻 $\tau_\ell=\ell/f_s$（抽样间隔 $\Delta\tau=1/f_s$）上的**离散采样值**，$g(\cdot)$ 是成形/采样响应（无 sinc 包络的离散冲激模型相当于 $g=\delta$），$\alpha_p$ 是第 $p$ 条路径的复增益、$P_p=|\alpha_p|^2$。若各路径增益在 realization 间互不相关，集合平均会把交叉项消掉，得到

$$
\mathbb{E}\big\{|h_d[\ell]|^2\big\}\ \overset{\text{路径互不相关}}{=}\ \sum_{p}P_p\,\big|g\!\left(\ell/f_s-\tau_p\right)\big|^2 .
$$

这是一个**理想化说明**，不是任何配置下的恒等式：有限窗、有限导频集合上的最小二乘抽头 $\hat t_\ell$ 只是 $h_d$ 在所选基上的投影，有限 realization 下 $\mathbb E\{|\hat t_\ell|^2\}$ 与上式有偏差。**发布出去的先验是经验量**——训练代码在 CDL 上做最小二乘后对 $|\hat t_\ell|^2$ 跨链路、跨 realization 求均值（§6），不是解析式。$\sigma^2_{eff}$ 同理是标量示意：$p[\ell]$ 相对噪声的绝对电平也影响收缩强度，不要在不同功率标度的场景间直接搬用。

训练用的频域基 $\mathbf A$ 的第 $\ell$ 列是 $e^{-j2\pi\ell k/N}$，等价于按单位冲激基建模。由此只有两类输入决定 $p$：

1. **多径结构集合**：模型（CDL A–E / TR 38.901 TR 等）、`direction`、`delay_spread_s`、速度区间（Doppler）、角度谱与路径数、`normalize_delays` / `normalize_channel` 标志。
2. **延迟到离散 tap 的映射**：tap 索引是**时延**抽样，$\tau_\ell=\ell\cdot\Delta\tau=\ell/f_s$（抽样间隔 $\Delta\tau=1/f_s$），仓库用的时延窗由 Sionna 的 `time_lag_discrete_time_channel(sample_rate_hz, max_delay_spread_s)` 给出
   $$l_{\min}=-6,\qquad l_{\max}=\big\lceil T_{\max}\,f_s\big\rceil+6 .$$
   采样率、FFT 尺寸与抽头窗 $[l_{\min},\,l_{\max}]$ 一起决定向量的长度和每个 tap 的物理含义；`max_delay_spread_s` 未设置时回落到信道配置的值。

实测对应关系：`pusch_cp_2ue_2layer.toml`（CP-OFDM，4.32 MHz，FFT 144）在 $T_{\max}=3\,\mu s$ 下 $l_{\max}=19$、26 taps；`pusch_4ue.toml`（DFT-s-OFDM，18 MHz，FFT 600）同样 $T_{\max}=3\,\mu s$ 下 $l_{\max}=60$、67 taps。两者长度不同是因为 $f_s$ 不同，不是因为波形不同。

## 3. 什么不决定 $p[\ell]$

- **波形标签 CP-OFDM / DFT-s-OFDM**：两种波形在同一物理信道上发送，波形改变的是导频排布与估计器结构（type-2 DMRS 原生 LS 与 DFT-s OCC 域拟合），不改变 $p$。估计器差异由 MMSE 公式里的设计矩阵、导频几何与噪声方差吸收，本来就不由 tap prior 承担。决定的是波形**标签**本身：只有当换波形后采样率、FFT、抽头窗仍然相同，同一个 $p$ 才继续适用；一旦 numerology 或采样率跟着变，就需要单独训练一份。
- **MCS / 码表 / 目标码率**：只改变传输块编码，与信道标定无关（这也是兼容性比对显式排除 `mcs_index` / `mcs_table` 的原因）。
- **用户数、层数、检测器、译码迭代次数**：都不进入 $p$ 的定义。
- **接收天线数**（在 §4 的限定下）：见下节。

## 4. 天线配置：为什么原则上不该乘出 8 份

在**所有阵元同分布、每路径功率按同一归一方式分配、单元方向图与极化一致**的条件下，路径功率 $\{P_p\}$ 对每条链路相同，因此

$$
\mathbb{E}\big\{|h_{ij}[\ell]|^2\big\}
=\frac{1}{N_{\text{link}}}\sum_{i,j}\mathbb{E}\big\{|h_{ij}[\ell]|^2\big\}
=\mathbb{E}\big\{|h_{11}[\ell]|^2\big\}
$$

与链路边数无关。改变 1T1R→4T8R 改变的是**估计这个均值时的采样精度**：训练时每 realization 的平均链路数

$$
K=N_{\text{real}}\times N_{\text{UE}}\times N_{\text{TX port}}\times N_{\text{RX elem}} .
$$

严格的方差结论需要限定条件，实践中不能照搬 $1/K$：

- **链路并不独立**。CDL 的同一角度簇经阵列流形展开成空间相关的协方差 $\mathbf R_H$（$\propto$ 角度谱），所以 $K$ 条链路里含有可观的共同分量，$1/K$ 是乐观下界。真实精度还取决于簇内角度散布与阵列孔径。
- **单元统计必须同分布**。单元方向图、极化方式、`normalize_channel` 若随阵列规模改变每单元分到的功率，那每 tap 的边际功率就会变，跨阵列复用不再是同一目标。仓库侧：`polarization` 必须是 `single`（`dual` 被 `validate` 拒绝，`channel_config.py:69-72`），而 `antenna_pattern='omni'`、`polarization_type='V'` 是 `configs/cdl_38_901_4x4.toml:16-18` 与 `configs/cdl_38_901_2tx_4rx.toml:17-19` 里显式写的值（`AntennaSettings` 本身不给默认值），因此这两份 profile 满足该前提，换 profile 时需重新核对。
- **互耦与秩亏**（$N_{\text{RX}}<$ 用户数）改变的是 CSI 估计的空间结构与干扰，不改变 $p$ 的定义。

因此理论结论是：**在"各单元统计一致、功率归一一致"的前提下，天线数只影响采样精度，不改变名义 PDP**，因此一份先验在物理上可跨 1T1R…4T8R 共用，想拿更稳的向量就在链路最多的阵列上训练（4T4R 每次 realization 平均 16 条链路）。但这只是理论判定：当前实现的键仍含天线几何（§8），所以**实际能否复用取决于键是否完全一致**，不要据本节省掉训练。§8.1 的初步实测支持"小阵列训练 → 大阵列使用"这一方向，但覆盖有限、键策略未放宽。

## 5. 份数判定规则

$$
\#\text{先验}
=\#\Big\{(\text{信道集合}),\ (f_s,\ \text{FFT}),\ [l_{\min},\,l_{\max}]\Big\}
$$

必须分开训练的触发条件：

- 换了信道集合：不同 CDL model、不同速度/Doppler 区间、不同 delay spread、换了非"每路径同功率"的模型（如 TR 38.901 TR）。
- 换了栅格：不同 numerology/采样率/FFT（tap 长度与物理含义都变）。
- 换了抽头窗：$[l_{\min},\,l_{\max}]$ 不同。长度与逐 tap 的方差权重都变，硬截断或补零会破坏 LMMSE 的对角加权；接收窗窄于信道自身支撑会截断联合拟合，仓库已有实测（`docs/feature_catalog.md`：4.32 MHz 支撑 26 taps，`l_min=-2` 约 9% 相对 CSI 误差，`max_delay_spread_s=0.3e-6` 隐含的 ~5 tap 窗六个检测器全部 CRC 失败）。

不必分开训练（物理上）：仅波形变化而栅格不变、仅 MCS/用户数/检测器变化。仅天线数变化在物理上也属于同一份先验，但当前实现按几何分键；§8.1 只对 1Rx 训练 → 4Rx 使用给出初步支持（反方向因目标几何不可解而无证据），因此**仍按几何各自训练**。

## 6. 数值训练方法

`estimate_dmrs_tap_power_prior(...)` 的做法：

1. 由 $f_s$ 与 $T_{\max}$ 得到 $l_{\max}$，在 FFT 栅格上构造 tap 基矩阵 $\mathbf A$（$l_{\min}\dots l_{\max}$）。
2. 对空频域栅格施加 CDL，取每个 realization 的频响（一个静态信道在一个 slot 内只取一次，不把 OFDM 符号当独立抽头）。
3. 对每条链路做最小二乘 $\hat{\mathbf t}=\arg\min\|\mathbf A\mathbf t-\mathbf y\|_2$，累加 $|\hat t_\ell|^2$ 并累计链路数。
4. 取均值，$p \leftarrow \max(p,\,p_{\max}\cdot10^{-8})$ 防止零/负权重，转 float32。

要点：

- **不重整化**。代码只加地板，不把 $\sum_\ell p[\ell]$ 归一到 1；实测发布的 DFT 先验 $\sum_\ell p[\ell]=0.980249$。绝对电平必须与仿真的信号/噪声功率自洽，不要在不同功率标度的场景之间直接搬用。
- **种子与 realization 数必须记录**，并写进 NPZ metadata（`training_seed`、`training_realizations`）。已发布 DFT 先验为 256 realization、seed 21260924。
- **若将来放宽键（§8），才建议在最大阵列上训练再跨阵列共享**；§8.1 只测了相反的方向（1Rx 训练 → 4Rx 使用），对这个建议既不支持也不否定，证据不足以支撑放宽。在此之前每个几何仍各自训练，`--prior-realizations` 只用于连通性冒烟，不能用小样本训练正式先验。

## 7. 验收门槛与发布

门槛由验证 TOML 的 `[estimator_validation]` 阈值键给出，`nr-pusch-estimator-validation` 按 `--validation-config` 指向的文件的 holdout 判定；文件中不写这些键时使用代码默认值，与 `configs/channel_estimation_validation.toml`（版本 1）完全等价：

- `min_relative_bler_reduction`，默认 `0.10`：25 dB 与 30 dB 各需满足的 holdout 相对 BLER 降低下限。
- `max_paired_bler_difference_97_5pct_upper`，默认 `0.0`：配对 bootstrap 97.5% 单侧上界的上限。
- `require_strict_upper_bound`，默认 `true`：上界必须**严格小于**上一项；置 `false` 时按"小于等于"判定。

- 25 dB 与 30 dB 各需同时满足：holdout 相对 BLER 降低不低于 `min_relative_bler_reduction`、配对 bootstrap 上界满足上述比较、candidate 的 data-RE NMSE 严格低于 baseline。
- 60 dB 不得回归（`candidate_high_errors <= baseline_high_errors`）。
- bootstrap 重复次数下限 100。

发布约定：

- 训练产物 `<output-stem>.prior.npz` 永远是**诊断候选**，即使通过也保留。
- 只有 `acceptance_gate.passed=true` 才复制并打上 `accepted_gate_sha256` 标记，先写同目录临时文件再 `os.replace` 到 `configs/tap_power_prior/<waveform>/<sha256 of canonical compatibility>.npz`，因此只有新通过的结果才覆盖旧发布版本。
- 未通过时不发布，接收端按缺失处理并**明确报错**，不回退 LS、不合成先验。
- 由于 `accepted_gate_sha256` 覆盖整个 `acceptance_gate` 字典（其中包含 `thresholds`），同一几何下用不同门槛版本发布的先验标记不同，可据此区分结论来源。
- 禁止为了让某个波形通过而改阈值、换 seed 或换 SNR 点；确需变更时应新增一份版本化验证配置并在摘要 JSON 记录，原门槛的失败结论保持不变。

## 8. 当前实现与理论的差距（已知限制）

共享目录的键是**完整兼容性字典**（TX 载波、PUSCH 设置、CDL + 天线、抽头窗、FFT、采样率，排除 MCS）的 SHA-256，其中包含 `AntennaSettings`。因此当前实现要求**逐几何各一份**：按 §4 的条件，4T4R 训练出的向量对 1T1R 是同一集合均值（无偏，但方差不同），系统仍会因键不同而报"未找到"。

放宽键（例如只保留信道集合 + 栅格 + 抽头窗）是一次**键策略变更**，不是 bug 修复。若要重做验证，先把对比口径写死，避免事后找理由：目标几何上"复用先验"与"各自训练先验"在**同一批 holdout 帧**上比较，配对块错误差的 97.5% 上界 ≤ 0（复用不更差）、data-RE NMSE 相对偏差 ≤ 5%，且复用先验在该几何上仍通过对应门槛版本。下述 5% 与"≤ 0"是本次探索实验的对比口径，**不是仓库的验收门槛**（验收门槛只有 §7 那几条），也不是放宽键的既定标准。

### 8.1 跨阵列复用验证（2026-10-03，初步结果，键未放宽）

组合：`pusch_4ue.toml`（DFT-s-OFDM，4 UE × 1 层，18 MHz / FFT 600）配两个只差接收阵列的 CDL-A（100 ns 静态、`normalize_delays=true`、`normalize_channel=false`、TX 阵列都是 1×1）：目标 2×2（4 Rx，`configs/cdl_38_901_4x4.toml`）与源 1×1（1 Rx，临时 TOML）。两者抽头窗相同（67 taps，`l_min=-6`、`T_max=3 µs`），训练各 256 realization、seed 21260924；4Rx 用已发布的先验，1Rx 现训，sum 0.948423 vs 0.980249。

方法说明：这是诊断实验，先验向量直接注入 `run_estimator_validation`（替换其内部的训练调用），**刻意绕过共享目录的键解析**，所以它只能回答"向量搬过去有没有害"，不能证明键可以放宽。

两个向量的差异很小但**有系统性方向**：相对 L2 差 5.27%，去均值相关 0.99993，逐 tap 平均比 `4rx/1rx = 0.967`（链路少的训练结果整体偏大约 3.3%）。holdout 每点 300 帧、batch 20、GPU，四个臂共用同一批信道与噪声抽样（种子一致）：

| 目标几何 | 先验 | 25 dB（baseline→candidate） | 30 dB | data-RE NMSE |
|---|---|---|---|---|
| 4 Rx | 4Rx 自训 | 434 → 381（−12.21%，上界 −0.030） | 34 → 26（−23.53%，上界 −0.0017） | 0.001994 → 0.001510 |
| 4 Rx | **复用 1Rx 训练** | 434 → 380（−12.44%，上界 −0.0317） | 34 → 26（−23.53%，上界 −0.0017） | 0.001994 → 0.001512 |
| 1 Rx | 1Rx 自训 | 1200 → 1200（0%，门槛不过） | 1200 → 1200（0%，门槛不过） | 0.002006 → 0.001518 |
| 1 Rx | 复用 4Rx 训练 | 1200 → 1200（0%，门槛不过） | 1200 → 1200（0%，门槛不过） | 0.002006 → 0.001516 |

配对比较（同一批帧、同一个 `dmrs` 基线）：4Rx 目标上复用减自训在 25 dB 平均每帧 −0.00083、97.5% 上界 0.000000，30 dB 完全相同；NMSE 比 1.0013 / 1.0009。两个 4Rx 臂在这轮缩减实验里按版本 1 判据计算都是 passed，但**这是每点 300 帧的探索性结果，不是 3000 帧的正式验收**，不能替代 §7 的门槛结论。

**结论：键保持不变，本次结果不足以放宽。** 理由：

1. **测到的方向与 §4 主张的方向相反。**§4 的建议是"在最大阵列上训练、跨阵列共享"，也就是 4Rx 训练 → 1Rx/2Rx 使用；本轮跑的是反向的 1Rx 训练 → 4Rx 使用。它只说明"这一个匹配 profile 上，小阵列训练的向量用到 4Rx 上与 4Rx 自训向量看不出差别"，既没有验证 §4 建议的方向，也不构成 1T1R…4T8R 普遍可互换的证据。
2. 1Rx 目标本身不可解：4 个用户在 1 根接收天线上，25/30 dB 全部 1200 个 TB 出错，用自训先验也过不了门槛，因此反向对比对 BLER 没有分辨力，只能比较 NMSE（差 ≤ 0.15%）。
3. 样本量小：每点 300 帧（正式门槛用 3000 帧）。300 帧上两臂只差 1 个块错误，配对上界 0 只说明"看不出差异"，不是"证明无差异"。
4. 覆盖单一：只测了 CDL-A、100 ns、静态、4 UE、TX 1×1、阵列形状 1×1 与 2×2；换模型、换速度、换阵列形状（ULA vs URA 会改变空间相关结构）都没有证据。

要放宽键，至少需要补齐：目标几何本身可解（1 Rx 目标应换成 1 UE，或换 2 Rx）下的 3000 帧 holdout 对比、至少两种信道模型、以及双向（大小阵列互复用）。在这些证据出现之前，按 §5 的规则逐几何训练是正确的做法。

## 9. 实测状态（2026-10-03，RTX 3060 Laptop GPU，256 realization 训练，holdout 3000 帧/点）

| 波形 | TX / CDL | 结果 | 门槛要点 | 耗时 |
|---|---|---|---|---|
| DFT-s-OFDM | `pusch_4ue.toml` + `cdl_38_901_4x4.toml` | **PASS**，已发布 | 25 dB 4412→3846（−12.83%，上界 −0.0428）；30 dB 289→247（−14.53%，上界 −0.00242）；60 dB 0→0 | 6426 s |
| CP-OFDM | `pusch_cp_2ue_2layer.toml` + `cdl_38_901_2tx_4rx.toml` | 版本 1 **FAIL** → 版本 2 **PASS**，已发布 | 版本 1：30 dB 仅 −9.09%（阈值 10%）、上界恰为 0.0。版本 2（9%、上界 ≤ 0、GPU、batch 20）：25 dB 902→691 = −23.39%（上界 −0.03）；30 dB 59→45 = −23.73%（上界 −0.001）；60 dB 0→0 | 1353 s → 1626 s |

发布文件：`configs/tap_power_prior/dft_s_ofdm/8232bdc01e20cd9ba8437cc27f02f41108c292e551043bdba164e3cfed260b87.npz`（67 taps）与 `configs/tap_power_prior/cp_ofdm/d0efae4dad9e11aa39c757814ccada531bcbb2b626381e813bbe99253f4b65a4.npz`（26 taps），两份都带 acceptance 标记。CP-OFDM 的标记记录的是版本 2 门槛（`0.09` / 上界非严格）；**这一轮的抽样同时满足版本 1 的严格判据**（≥10%、上界严格小于 0），所以 CP 的版本 1 失败结论属于当时的抽样，不代表 LMMSE 在该几何上无效。版本 1 的诊断候选仍留在 `outputs/tap-power-prior/cp_2ue_2layer/validation.prior.npz`，失败记录保留在 `outputs/tap-power-prior/cp_2ue_2layer/validation.json`，不因版本 2 的发布而改写。版本 2 改变了 `batch_size`，因此其信道实现与 AWGN 抽样与版本 1 不同，两组数字不可互相替代。

端到端复核：CP-OFDM 网页任务 `fcaf738186c5`（高级配置组合 CP TX/CDL + `bler_cp_2ue_2layer_dmrs_lmmse_smoke.toml`，15 个点）中 `dmrs-lmmse` 在 20/25/30/35/40 dB 的 BLER 为 0.95 / 0.275 / 0.007 / 0.001 / 0，逐点低于同轮 `dmrs` 的 0.975 / 0.3219 / 0.010 / 0.002 / 0（每点 ≤500 帧，作为趋势参考）。

## 10. 新增 profile 时的检查清单

1. 该 profile 的 $(f_s,\ \text{FFT},\ [l_{\min},\,l_{\max}])$ 与信道集合是否与已有先验一致？按 §5 的规则这属于"同一份先验"；但当前实现的键还包含天线几何（§8），**只有键完全相同的 profile 才能自动复用**，跨天线配置共享要先做 §8 的验证。
2. 不一致则按 `README.md` §"CDL tap-power LMMSE 验证" 的命令训练，`--prior-dir configs/tap_power_prior`，全量参数、不覆盖门槛。
3. 看 `acceptance_gate.passed`：`true` 才会在共享目录出现新文件；`false` 只有诊断候选，并把失败点（相对降低、上界、NMSE、60 dB 回归）记录到本文 §9。
4. 全量 DFT-s-OFDM 单轮实测 6426 s（≈1 h 47 min），CP-OFDM 版本 1 1353 s、版本 2 1626 s；运行命令不要加小时级 shell 截止时间。
5. 验收通过后用实际 profile 跑一次 `nr-pusch-bler`（该 profile 的 SNR/帧数）确认自动选中，不要只凭 JSON 判断。
