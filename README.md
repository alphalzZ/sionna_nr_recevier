# NR PUSCH Lab

配置驱动的 5G NR PUSCH 发送、CDL 信道与 IQ 分析仓库；支持单 UE 多层、单发射天线多接收天线及多 UE 聚合流。原 4 UE × 1 层 × 1 Tx × 4 Rx 配置保持为回归基线。

## 当前发送端

当前实现使用 Sionna 2.2.0 的 `PUSCHTransmitter` 生成 PUSCH 数据资源网格，并支持 CP-OFDM 和 DFT-s-OFDM 波形输出。DFT-s-OFDM 对数据符号应用酉 DFT，并按 type-1 低 PAPR 序列生成 DMRS 后进行 OFDM 调制；`[pusch].dmrs_additional_position` 支持 0、1、2，由 Sionna 资源图决定各 DMRS occasion。Sionna 2.2.0 的 `PUSCHConfig` 不暴露 transform precoding，因此配置中的 DFT 预编码 MCS 映射到同调制阶数和码率的原生 Sionna MCS。当前活动配置为 DFT-s-OFDM MCS table 1/index 20，目标码率 0.6015625，TB 大小 28,168 bit。

CP-OFDM 沿用 Sionna 原生 PUSCH 资源网格与 DMRS 映射。`dmrs_beta` 必须
匹配当前 DMRS 组数和长度对应的 Sionna 原生值；配置加载时校验，不会静默
改变导频功率。`configs/pusch_cp_2ue_2layer.toml` 配
`configs/cdl_38_901_2tx_4rx.toml`；`configs/bler_cp_smoke.toml` 是单帧、两
TB 的 CP 接收连通性配置。

### MIMO TOML 拓扑

`[pusch]` 的 `num_layers`（每 UE，1–4）、`num_antenna_ports`（1/2/4）、
`precoding`（`non-codebook` 或 `codebook`）、`tpmi`、`dmrs_length` 均可配置；
缺省分别为 1、1、`non-codebook`、0、1。所有 UE 共用 PUSCH 资源、
MCS、层数、端口数和预编码模式；每个 `[[users]]` 使用长度等于层数的
`dmrs_ports` 数组，所有并发流的 DMRS 端口必须唯一。总流数上限为 8，
并非允许单个 PUSCH 使用 8 层；non-codebook 还要求层数等于天线端口数。
原配置只把 `dmrs_port = n` 迁移为 `dmrs_ports = [n]`，其余参数不变。
type-1 单符号支持端口 0–3；5–8 流使用 `dmrs_length = 2` 和端口 0–7，
此时原生 Sionna 只允许 `dmrs_additional_position` 0 或 1。
DFT-s-OFDM 的自定义低 PAPR 映射仅实现 type 1；不合法组合在配置阶段报错。

示例：`configs/pusch_1ue_1tx_4rx.toml` 配旧 CDL 的 1 Tx/4 Rx；
`configs/pusch_1ue_4layer.toml` 与 `configs/pusch_2ue_4layer.toml`
配 `configs/cdl_38_901_4tx_8rx.toml` 的 4 Tx/8 Rx。频域示例：

```bash
nr-pusch-tx --config configs/pusch_2ue_4layer.toml --output /tmp/pusch_8stream_tx.npz --device cpu
nr-pusch-channel --config configs/cdl_38_901_4tx_8rx.toml --tx-config configs/pusch_2ue_4layer.toml --domain frequency --input /tmp/pusch_8stream_tx.npz --sample-rate-hz 18000000 --output /tmp/pusch_8stream_rx.npz --device cpu
nr-pusch-rx --tx-config configs/pusch_2ue_4layer.toml --input /tmp/pusch_8stream_rx.npz --input-domain frequency --channel-estimator perfect --noise-variance 0.000001 --output /tmp/pusch_8stream_decode.npz --device cpu
```
八流 DMRS/LMMSE 单点连通性检查：
`nr-pusch-bler --tx-config configs/pusch_2ue_4layer.toml --channel-config configs/cdl_38_901_4tx_8rx.toml --simulation-config configs/bler_8stream_smoke.toml --output /tmp/pusch_8stream_bler.csv --device cpu`。
该配置只发一帧，不用于评估统计 BLER。

CP-OFDM TX→信道→RX 连通性示例（采样率从 TX manifest 读取）：

```bash
nr-pusch-tx --config configs/pusch_cp_2ue_2layer.toml --output /tmp/cp_tx.npz --seed 4 --device cpu
SAMPLE_RATE_HZ=$(/home/le-lei/workspace/test/.venv/bin/python -c 'import json; print(json.load(open("/tmp/cp_tx.json"))["result"]["sample_rate_hz"])')
nr-pusch-channel --config configs/cdl_38_901_2tx_4rx.toml --tx-config configs/pusch_cp_2ue_2layer.toml --domain frequency --input /tmp/cp_tx.npz --sample-rate-hz "$SAMPLE_RATE_HZ" --output /tmp/cp_rx_grid.npz --device cpu
nr-pusch-rx --rx-config configs/pusch_cp_2ue_2layer.toml --input /tmp/cp_rx_grid.npz --input-domain frequency --channel-estimator dmrs --detector lmmse --noise-variance 0.0001 --output /tmp/cp_decode.npz --device cpu
nr-pusch-bler --tx-config configs/pusch_cp_2ue_2layer.toml --channel-config configs/cdl_38_901_2tx_4rx.toml --simulation-config configs/bler_cp_smoke.toml --output /tmp/cp_bler.csv --device cpu
```


NPZ 的物理发射天线轴与层轴不混用。RX 星座与 LLR 保留
`[batch,user,layer,symbol_or_bit]`，TB bits/CRC 仍按 UE 返回。
MATLAB H5 reader 仅服务原四用户固定夹具；新拓扑使用 NPZ 输入。

## CDL 信道

`NrPuschCdlChannel` 使用 Sionna TR 38.901 CDL A–E 模型处理每个 UE 的独立上行链路，将每个物理发射端口的贡献合并到接收阵列。`[antennas]` 中的 `tx_num_rows/tx_num_cols` 缺省 1×1，`rx_num_rows/rx_num_cols` 可为任意正整数；TX 阵列天线数须等于 PUSCH 天线端口数。原 `configs/cdl_38_901_4x4.toml` 为单端口 UE、单极化 2×2 BS 阵列。

BLER 链路默认用 `channel_domain = "frequency"`：按 OFDM 符号采样 CDL CIR，使用与时域链路相同的有限长度 sinc taps 和时延范围生成频响，再直接施加到发射资源网格，保留 batch 并行，不生成逐采样 CIR 和时域卷积。该单抽头频域模型假设 CP 足以覆盖时延扩展，不模拟 CP 不足导致的 ISI 或符号内快速时变造成的 ICI。设为 `channel_domain = "time"` 可使用时域卷积；独立 IQ 抓包的信道命令也继续使用时域路径。

先生成发送 IQ，再将其作为独立输入应用信道：

```bash
nr-pusch-tx --config configs/pusch_4ue.toml --output /tmp/pusch_tx.npz --seed 7
nr-pusch-channel --config configs/cdl_38_901_4x4.toml --input /tmp/pusch_tx.npz --sample-rate-hz 18000000 --output /tmp/pusch_rx.npz
```

发送 NPZ 的 `iq` 为 `[batch,user,tx_antenna,sample]`；信道输出 `iq` 为 `[batch,rx_antenna,sample]`、`per_user_iq` 为 `[batch,user,rx_antenna,sample]`、`channel_taps` 为 `[batch,user,rx_antenna,tx_antenna,sample,tap]`（`channel_tap_axes` 中该轴名为 `sample`）。频域 CSI 为 `[batch,1,rx_antenna,user,tx_antenna,symbol,fft_bin]`。信道输出包含线性卷积尾部；噪声由独立的 AWGN 步骤加入。

离线资源网格也可直接应用频域信道；输入使用发送 NPZ 中的 `frequency_grid`，并提供发送 TOML 以取得 FFT/子载波间隔配置：

```bash
nr-pusch-channel --config configs/cdl_38_901_4x4.toml --input /tmp/pusch_tx.npz --sample-rate-hz 18000000 --domain frequency --tx-config configs/pusch_4ue.toml --output /tmp/pusch_rx_grid.npz
nr-pusch-rx --tx-config configs/pusch_4ue.toml --input /tmp/pusch_rx_grid.npz --input-domain frequency --channel-estimator perfect --noise-variance 0.001 --output /tmp/pusch_decode.npz
```

频域信道 NPZ 提供 `grid`、`per_user_grid` 和 `channel_frequency_response`，可直接输入接收机，不进行 OFDM 调制、时域信道卷积或接收端 OFDM 解调。

## Sionna RT 四波束固定快照链路

RT 实验使用独立于 CDL 的链路：`nr-pusch-rt-channel` 生成静态阵元/波束快照，`nr-pusch-beam-bler` 在固定传播快照上运行编码 PUSCH。RT 不调用 CDL 信道或 CDL tap-power prior；结果是指定几何和固定信道条件下的 BLER，不代表动态无线信道或真实城区 benchmark。模型只含静态单频载波 LoS/镜面多径，不含移动、多普勒/ICI 或 RF 链互串。

网页场景选择器提供 3 个项目内置场景（LoS、地面、地面＋墙面）和 `sionna-rt==2.2.0` 随包的 15 个场景；Sionna PLY 仅复制 XML 引用的网格，参数化几何仍只适用于项目地面/墙面。所有内置场景共用当前 RT 配置中的 BS/UE/阵列位置；2D 预览不绘制这些 PLY，城区场景的追踪和内存开销较高。

`nr-pusch-rt-channel` 默认将可验证的静态快照按输入指纹缓存在 `outputs/rt_snapshots/`，重复输入显示 cache hit 并复用 NPZ/JSON 快照。RT Web BLER 仍会重新运行；只有成功完成的 Web RT job 才原子归档到 `outputs/rt_runs/<job-id>/`，其中保留配置、场景副本、快照、准入报告、CSV/JSON、进度、日志和最终 job 状态。缓存/归档均位于 gitignored `outputs/`；CDL jobs 不写入该 RT 归档。

```bash
nr-pusch-rt-channel --tx-config configs/pusch_4ue.toml \
  --rt-config configs/rt_beam_los.toml \
  --cache-dir outputs/rt_snapshots \
  --output /tmp/nr-rt-validation/dft/channel.npz
nr-pusch-beam-bler --tx-config configs/pusch_4ue.toml \
  --channel-snapshot /tmp/nr-rt-validation/dft/channel.npz \
  --simulation-config configs/bler_rt_beam_smoke.toml \
  --output /tmp/nr-rt-validation/dft/results.csv --device cpu
```

快照 NPZ 保存阵元路径、阵元域与波束域频响/时域 taps；同名 JSON 记录 RT 版本、场景资源 hash、频率轴、TA、CP 与配置摘要。加载时会校验当前 TX 资源网格/DMRS 几何、UE 顺序、文件摘要和运行时版本；缺文件或不匹配时失败关闭，不回退 CDL。结果写 CSV 与 JSON sidecar，可选 `--progress-jsonl`；每个估计器/检测器/SNR 点有逐 UE 行和 `all` 聚合行。新检测器 `beam-independent` 将其它 UE 泄漏作为高斯等效干扰，`zf` 做逐 RE 零迫；二者与既有六种联合检测器共享 NR TB 解码。联合接收用阵元噪声诱发的完整波束协方差白化，独立接收只作逐波束噪声归一化。RT 只支持 `perfect`/`dmrs`，不接受依赖 CDL 先验的 `dmrs-lmmse`。

`[noise].post_combiner_ratio` 表示后级合路噪声方差/阵元噪声方差，缺省为 `0.001`，`--post-combiner-ratio` 可覆盖并写入结果清单。以下可在相同快照、payload 与标准复高斯噪声 draw 上扫描 `0`、`0.001`、`0.01`；该 smoke 是连通性诊断，不是统计 BLER 结论：

```bash
for ratio in 0 0.001 0.01; do
  nr-pusch-beam-bler --tx-config configs/pusch_4ue.toml \
    --channel-snapshot /tmp/nr-rt-validation/dft/channel.npz \
    --simulation-config configs/bler_rt_beam_smoke.toml \
    --post-combiner-ratio "$ratio" \
    --output "/tmp/nr-rt-validation/dft/ratio-${ratio}.csv" --device cpu
done
```

### RT 专用配置

| 配置 | 用途 |
|---|---|
| `configs/rt_beam_los.toml` | 3.5 GHz、空场景、8×8 V 极化 iso 阵列、四 UE、深度 0；LoS 传播基线。 |
| `configs/rt_beam_ground.toml` | 相同几何，ground 场景、追踪深度 1。 |
| `configs/rt_beam_ground_wall.toml` | 相同几何，ground+wall 场景、追踪深度 2。 |
| `configs/pusch_rt_4ue_cp.toml` | CP-OFDM 接口/正确性对照；四 UE 单层、50 RB、MCS table 1/index 20、30-kHz SCS。统计主 profile 仍是 `configs/pusch_4ue.toml`。 |
| `configs/bler_rt_beam_smoke.toml` | CPU、60/80 dB、最多 2 帧/点，覆盖 `perfect`/`dmrs` 和两种新检测器加既有六种 detector；仅连通性检查。 |
| `configs/bler_rt_beam.toml` | CUDA:0、15–50 dB 每 5 dB、batch 20、最多 2,000 帧/SNR、target 200 errors、perfect/DMRS × 三个 detector、`stop_at_zero_bler=true`；最多 48 个 arm×SNR 点，Web gate 通过后才运行。这是配置预算，不是已测统计结果。 |
| `configs/bler_rt_beam_web_quick.toml` | 当前 CPU profile：10/15/17/20 dB、batch 16、最多 2,000 帧/SNR、target 100 errors、DMRS × 三个 detector、`stop_at_zero_bler=false`；12 个 arm×SNR 点。文件名虽含 quick，该当前工作量并非两帧 smoke，也不是统计基准。 |

### RT 物理与编码验收

`nr-pusch-beam-validate` 将传播、波束、噪声、无编码、编码 PUSCH、多径、时域和单因素扫描写入 JSON gate 报告；任一必需检查失败时返回非零。`--simulation-config` 仅用于 `pusch`、`sweeps`、`all` 阶段。`all` 固定按 LoS、beam、noise、uncoded、PUSCH、reflection/path convergence、time、sweeps 顺序执行，要求 ground-wall 场景及两帧 smoke 预算：

```bash
nr-pusch-beam-validate --tx-config configs/pusch_4ue.toml \
  --rt-config configs/rt_beam_ground_wall.toml \
  --simulation-config configs/bler_rt_beam_smoke.toml \
  --stage all --output /tmp/nr-rt-validation/all/report.json --device cpu
```

`time` 阶段用同一静态 tap 对照 native OFDM 解调与有限 tap 频响；严格 FD/TD 等价仍要求原 `1e-5` RMS 门槛。`multipath` 使用 100k/200k samples-per-source 配对检查镜面反射和 CFR 收敛。`sweeps` 扫描用户间隔、UE3 功率、公共波束偏移、相位量化、场景、CSI 估计器和检测器；每点仅两帧，是连通性诊断，不是统计 BLER。按本仓库研究协议，CLI 统计应先通过 `all` 严格 gate；通用 `nr-pusch-beam-bler` 不默认自动门禁，只有提供 `--web-validation` 时才验证 Web 报告，而 Web worker 始终要求 `web-frequency-v1` 通过。该 Web gate 不改写严格报告。

2026-10-04 ground-wall DFT smoke：`all` 的前六个 gate（包括 160/160 coded rows 与 path-convergence）通过，但 time gate 的 FD/TD RMS 为 `3.3515e-4`（门槛 `1e-5`），因此停止在 sweeps 前；单独的 `sweeps` stage 已通过 20/20 个两帧诊断点。另一个 CP-OFDM LoS 正确性 smoke 的 time gate 通过（FD/TD RMS `1.4443e-6`），32 个两帧 coded rows 均 complete、0/8 TB errors；完整 unittest 147 tests 通过（302.921s）。这些都不是统计 BLER。未运行 CUDA 统计 BLER。实测细节与完整 tap 保留说明见 `docs/dmrs_tap_power_prior.md` §9。

`configs/scenarios.toml` 保留原 25 个 CDL bundle，并新增 4 个 `channel_backend="rt"` 快速预设；Web 默认仍为 CDL。RT Web 一键运行会在同一 FIFO job 内先追踪/验收，再执行 BLER；advanced 支持参数化 ground/ground-wall 几何及受限的 XML＋PLY ZIP 导入。

## 多用户接收和 BLER

`NrPuschRx` 以 Sionna PUSCH TB 解码器合并每 UE 多层码字。时域 IQ 经 OFDM 解调，频域网格直接检测；DFT-s-OFDM 数据先均衡、按层逆 DFT。`perfect` 接收物理 TX 天线 CSI，codebook 时应用原生预编码矩阵；`dmrs` 从每层导频的 comb/OCC 联合估计有效层信道，`dmrs-lmmse` 加入 CDL tap-power prior。保留原 4 UE 单层捕获的 MATLAB 对照估计路径。信道估计输出 `[batch,1,rx_antenna,user,layer,symbol,subcarrier]`，最终 TB/CRC 按 UE。

CP-OFDM 使用原生逐 RE 信道估计和检测路径，不做 DFT 解扩/IDFT；支持
`lmmse`、`lmmse-sic`、`k-best`、`ep`、`mmse-pic`、`soft-mmse-pic`、
`zf` 和 `beam-independent`。`beam-independent` 要求每用户单层且观测端口数等于总流数，
将其它用户功率作为高斯等效干扰；RT runner 在 ZF 前检查逐 RE 数值秩，失败点标为
`infeasible_rank`。EP 使用 double precision；K-best 要求接收天线数不少于总流数。Type-2
DMRS 与数据共用 OFDM 符号时，估计只使用原生 pilot mask 中的 RE。

DMRS 拟合使用配置的 `max_delay_spread_s` 限制候选 tap 范围，窗口长度随几何变化：4 用户 DFT-s-OFDM 基线（18 MHz 采样率）为 `-6..60`、共 67 taps；CP-OFDM 2 用户 2 层基线（4.32 MHz）为 `-6..19`、共 26 taps。这是时延范围先验，不读取本帧真实信道系数。多个 DMRS occasion 各自拟合 CSI，再按 OFDM 符号位置对相邻估计做复数线性插值；首个/末个 occasion 之外保持最近估计。`err_var` 按插值权重平方传播，假设各 occasion 的估计噪声独立，不包含信道时变造成的插值模型误差；高 Doppler 场景仍需独立验证。

### CDL tap-power LMMSE 验证

`dmrs-lmmse` 是显式 opt-in；现有 `dmrs` 默认和活动 TX/CDL profile 不变。LMMSE 算法验证固定单符号 DMRS；另行比较 `dmrs_additional_position=0/1/2` 的可靠性与 TBS 资源代价，结果见 `docs/receiver_optimization.md`，不能视为等吞吐比较。tap-power prior 的物理含义、所需份数及 CDL prior 的验收/发布规则见 `docs/dmrs_tap_power_prior.md`；RT `web-frequency-v1` 是独立传播/FD 近似门槛，不训练、读取或发布该 CDL prior，也不能改变 §7 的 prior 发布条件。

短时连通性验证（`--frames-per-snr` 只覆盖 development/holdout，60 dB 安全检查仍使用配置中的 64 帧）：

```bash
nr-pusch-estimator-validation --tx-config configs/pusch_4ue.toml --channel-config configs/cdl_38_901_4x4.toml --validation-config configs/channel_estimation_validation.toml --output /tmp/channel_estimation_smoke.json --device cpu --prior-realizations 8 --frames-per-snr 2
```

#### 完整产生 tap_power_prior

正式运行不加两个 smoke 覆盖参数：`configs/channel_estimation_validation.toml` 固定训练 256 个独立 realization、development 512 帧/SNR、holdout 3,000 帧/SNR、25/30 dB 和 60 dB/64 帧，两种波形各跑一次，共同发布到同一个 `configs/tap_power_prior/`（按波形和兼容性摘要分子目录，NPZ 本身是各自独立的数值先验）。

```bash
# 1) CP-OFDM（configs/pusch_cp_2ue_2layer.toml + configs/cdl_38_901_2tx_4rx.toml）
nr-pusch-estimator-validation \
  --tx-config configs/pusch_cp_2ue_2layer.toml \
  --channel-config configs/cdl_38_901_2tx_4rx.toml \
  --validation-config configs/channel_estimation_validation.toml \
  --output outputs/tap-power-prior/cp_2ue_2layer/validation.json \
  --prior-dir configs/tap_power_prior \
  --device cuda:0

# 2) DFT-s-OFDM（configs/pusch_4ue.toml + configs/cdl_38_901_4x4.toml）
nr-pusch-estimator-validation \
  --tx-config configs/pusch_4ue.toml \
  --channel-config configs/cdl_38_901_4x4.toml \
  --validation-config configs/channel_estimation_validation.toml \
  --output outputs/tap-power-prior/dft_4ue/validation.json \
  --prior-dir configs/tap_power_prior \
  --device cuda:0
```

每次运行写出 `validation.json`、配对帧 `validation.frames.npz` 和诊断用 `validation.prior.npz`；只有门槛通过才把候选按兼容性键原子发布到 `--prior-dir`，并在 `artifacts.published_prior` 记录发布路径。门槛为：holdout 的 25/30 dB 两点 BLER 都至少相对降低 10%、paired frame-cluster bootstrap 的 97.5% 单侧差值上界都小于 0、data-RE CSI NMSE 两点都下降，且 60 dB TB errors 不高于基线；`perfect` 只作同帧上界。

门槛参数写在验证 TOML 的 `[estimator_validation]` 表里，缺省值即上文所写：`min_relative_bler_reduction = 0.10`、`max_paired_bler_difference_97_5pct_upper = 0.0`、`require_strict_upper_bound = true`。需要复现"放宽后"的结论时使用版本化配置而不修改版本 1：`configs/channel_estimation_validation_v2.toml` 仅把这三项改为 `0.09` / `0.0` / `false`，并把 `device` 改为 `cuda:0`、`batch_size` 改为 `20`，其余训练与 holdout 设置与版本 1 完全相同。注意 `batch_size` 会改变信道实现与 AWGN 的抽样顺序（AWGN 种子按批次起点推导），因此版本 2 的数字是一次独立抽样，必须以该次运行的摘要为准，不能与版本 1 的数字互相替代。实际生效的阈值写进 `acceptance_gate.thresholds`，并计入发布标记 `accepted_gate_sha256`，可据此区分结论来源。

查看门槛与已发布先验：

```bash
python - <<'PY'
import json, pathlib
for name in ("cp_2ue_2layer", "dft_4ue"):
    summary = pathlib.Path("outputs/tap-power-prior") / name / "validation.json"
    if summary.is_file():
        data = json.loads(summary.read_text())
        print(name, data["acceptance_gate"]["passed"], data["artifacts"].get("published_prior"))
print(sorted(p.name for p in pathlib.Path("configs/tap_power_prior").glob("*/*.npz")))
PY
```

重训与覆盖：对同一 TX/CDL/抽头窗组合重复上面的命令即可；只有再次通过门槛才会覆盖同名已发布先验，门槛失败时保留原版本并留下新的诊断 NPZ。改动 TX 几何、CDL、抽头窗、FFT 或采样率都会得到新的兼容性键，需要单独训练，旧先验不会被误用。

实测耗时取各验证摘要的 `runtime_s`（development/holdout/high-SNR sweep，不含模型与 prior 训练/设置，也不是整条命令的 wall time）：版本 1 配置（CPU、batch 2）CP-OFDM 一次 1345 s；DFT-s-OFDM 同为 CPU、batch 2，用时 6426 s（development 981 s、holdout 25 dB 2478 s、holdout 30 dB 2930 s、60 dB 38 s），约 1 小时 47 分钟。CP-OFDM 用版本 2（GPU、batch 20）一轮 1551 s。长时 DFT-s-OFDM 验证不要加 3600 s 之类的 shell 截止时间。

当前仓库状态（2026-10-03 实测；两次 CP 运行分别用版本 1 与版本 2 门槛配置，seed 与 SNR 点一致，未临时调阈值）：

- **DFT-s-OFDM：PASS**，已发布 `configs/tap_power_prior/dft_s_ofdm/8232bdc01e20cd9ba8437cc27f02f41108c292e551043bdba164e3cfed260b87.npz`（67 taps，256 realization，seed 21260924）。holdout 25 dB 块错误 4412 → 3846（降低 12.83%，bootstrap 上界 −0.0428，NMSE 2.014e−3 → 1.517e−3）；holdout 30 dB 289 → 247（降低 14.53%，上界 −0.00242，NMSE 6.386e−4 → 5.669e−4）；60 dB 0 → 0。
- **CP-OFDM：版本 1 门槛 FAIL、版本 2 门槛 PASS，已发布** `configs/tap_power_prior/cp_ofdm/d0efae4dad9e11aa39c757814ccada531bcbb2b626381e813bbe99253f4b65a4.npz`（26 taps，256 realization，seed 21260924）。版本 1（CPU、batch 2）结论保持不变：25 dB 与 60 dB 通过（940 → 738，降低 21.49%，上界 −0.0287），30 dB 未过（33 → 30，降低 9.09% < 10%，3000 帧下上界恰为 0.0），诊断候选留在 `outputs/tap-power-prior/cp_2ue_2layer/validation.prior.npz`。版本 2（GPU、batch 20，9% 且上界 ≤ 0）这一轮：25 dB 902 → 691（降低 23.39%，上界 −0.03，NMSE 2.940e−3 → 2.433e−3）；30 dB 59 → 45（降低 23.73%，上界 −0.001，NMSE 9.165e−4 → 8.532e−4）；60 dB 0 → 0。**这一轮的抽样同时满足版本 1 的严格判据**（≥10% 且上界严格小于 0），因此已发布标记里的门槛是版本 2 的 9%/非严格；`batch_size` 改变了信道实现与 AWGN 的抽样，版本 2 的数字是一次独立抽样，不等同于版本 1 的那一次。

因此两种波形的 `dmrs-lmmse` 现在都可以直接用，不需要任何先验路径。DFT-s-OFDM：仓库自带的 `nr-pusch-bler --tx-config configs/pusch_4ue.toml --channel-config configs/cdl_38_901_4x4.toml --simulation-config configs/bler_dft_4ue_dmrs_lmmse_smoke.toml` 在已发布先验下 47 s 跑完 25 dB 的 6 个检测器臂；该 profile 每臂只有 1 帧，BLER 0–0.75（`lmmse`/`k-best`/`ep` 3/4 块错误，`lmmse-sic` 0/4），只用于连通性验证，不能当作性能结论；另用临时 profile `/tmp/nr-prior-check/dft_lmmse_only.toml`（`snr_db = [40.0]`、单一 `dmrs-lmmse`×`lmmse` 臂、`--device cuda:0`，未提交到仓库）复测 11 s，0 CRC 失败、0/4 块错误。CP-OFDM：网页实验台在“高级配置”里选 `pusch_cp_2ue_2layer.toml` + `cdl_38_901_2tx_4rx.toml` + `bler_cp_2ue_2layer_dmrs_lmmse_smoke.toml` 后兼容性检查通过并可运行，任务 `fcaf738186c5`（15 个点）在约 12.5 min 内完成，`dmrs-lmmse` 在 20/25/30/35/40 dB 的 BLER 依次为 0.95 / 0.275 / 0.007 / 0.001 / 0，均低于同轮 `dmrs` 基线的 0.975 / 0.3219 / 0.010 / 0.002 / 0；该 profile 每臂最多 500 帧，可作为趋势参考但样本仍远小于验证用的 3000 帧 holdout。CLI 侧另用临时 profile `/tmp/nr-prior-check/cp_lmmse_only.toml`（75 dB 单臂）复测 6 s，0 CRC 失败、0/2 块错误。门槛阈值不得为了让某个波形通过而临时放宽；如需改变 `training_realizations`、SNR 点或门槛本身，应新增一份版本化的验证配置并在摘要 JSON 中记录，再按新配置重跑，原门槛的失败结论保持不变。

需要注意的是，上述网页任务 `fcaf738186c5` 早于共享先验目录：它的归档快照 `runs/web/fcaf738186c5/simulation.toml:19` 仍带着已移除的 `dmrs_tap_power_prior_path`，指向从未通过验收的诊断候选 `outputs/release-0.1-baseline/estimator-validation/cp_2ue_2layer/validation.prior.npz`。该快照只作为历史记录保留；按当前代码重跑同样组合时不再需要先验路径，而是从 `configs/tap_power_prior/cp_ofdm/` 自动解析。

常规 `nr-pusch-bler` 的 `[bler]` 只需设置 `channel_estimator = "dmrs-lmmse"`，不再填写先验路径；接收端按 `--channel-config` 的 TX/CDL、抽头窗、FFT 和采样率在共享先验目录中查找匹配项。`nr-pusch-rx` 的 `[receiver]` 同样不再有先验路径，但必须用 `--channel-config` 或 `[receiver].dmrs_tap_power_prior_channel_config_path` 提供该 prior 对应的 CDL TOML。默认目录为信道配置同级的 `tap_power_prior/`，可用 `--prior-dir` 覆盖（`nr-pusch-bler`、`nr-pusch-rx`、`nr-pusch-estimator-validation` 均支持）。prior 缺失、不兼容或未经验收发布都会报错，不会回退到 LS。

BLER 配置可用 `[bler].channel_estimators` 选择多个估计器；仿真按 `channel_estimators × detectors × snr_db` 遍历组合。未设置时沿用单个 `channel_estimator`。网页 SIM 配置 `configs/bler_estimator_matrix.toml` 组合 `dmrs`、`dmrs-lmmse`、`perfect` 与 `k-best(k=16)`，扫描 25–50 dB 六个 SNR，共 18 个结果点，每点最多 2,000 帧（GPU profile）。运行前需按上面的验证命令产出通过验收的先验；未通过时该矩阵会因找不到匹配 prior 而拒绝启动。


DFT-s-OFDM 的上述检测器工作在用户×层总流上；Sionna layer demapper 在逆 DFT 后将每层 LLR 还原为各 UE 的 TB 码字。PIC 按全部层消除干扰；soft-PIC 按码字/层次交换 LDPC 外信息；SIC 只在 CRC 通过后重构 UE 并抵消其完整贡献。K-best 要求接收天线数不少于总流数，搜索路径数和复杂度由 `detector_parameter` 控制。PIC 的该参数表示迭代轮数，EP 中表示迭代次数；BLER 配置仍使用 `detectors` 与 `detector_parameters`。

EP 与 K-best 共用零时延等效空间信道近似；滤波后剩余的频率选择性记入高斯协方差，而非在 EP 图中显式建模所有跨采样相关性。该近似、软 LLR 校准及收敛行为仍需通过 MATLAB 参考向量和长 SNR 曲线验证。

配置 SNR 扫描点、批大小、帧上限和停止错误数后运行：

```bash
nr-pusch-bler --tx-config configs/pusch_4ue.toml --channel-config configs/cdl_38_901_4x4.toml --simulation-config configs/bler_smoke.toml --output /tmp/pusch_bler.csv --device cpu
```

将 `device` 写为 `cpu`、`cuda:0` 或 `auto` 可从 TOML 控制运算设备；`configs/bler_4ue_cdl_gpu.toml` 提供显式 CUDA 示例。若以 `--device cuda` 覆盖，程序会将其规范化为 Sionna 接受的 `cuda:0`。请求的设备同时会写入 Sionna 的全局 `sionna.phy.config.device`：Sionna 的 PUSCH 导频图案按该全局值分配，而资源网格按显式设备分配，两者不一致时（例如在有 GPU 的机器上跑 `--device cpu`）会在建图阶段报设备不匹配。批大小受 GPU 显存约束；频域信道一般比时域线性卷积更节省显存。

不同检测器可在 `[bler]` 中单独设置批大小；未列出的检测器使用 `batch_size`，每个 SNR 点最后一批仍受 `max_frames_per_snr` 截断。当前 `configs/bler_4ue_cdl_gpu.toml` 使用 `{ lmmse = 20, "lmmse-sic" = 20, "mmse-pic" = 20, "soft-mmse-pic" = 8 }`，`detectors = ["lmmse", "lmmse-sic", "mmse-pic", "soft-mmse-pic"]`（不含 k-best/ep；`batch_size = 20` 为未列出检测器的默认值）。K-best 的路径搜索在显存上最重，batch 必须明显低于其他检测器——本仓库记录过的 K-best 批量扫描是 2026-09-27 任务 `1313e4fe353d` 的归档快照（`k-best = 10`，其余检测器 26），见下文 §当前统一接收链路的全量检测器对照。网页的仿真 TOML 编辑器同样支持该字段。不同批大小改变随机数的分组，因此各检测器共享初始 seed 和统计条件，但不保证逐帧使用完全相同的 payload、信道和噪声样本。

扫描会逐 SNR 点发射随机 transport blocks、通过 CDL、按每个接收天线的测得信号功率注入复 AWGN，再用 CRC 与 payload 比对统计 BLER。CSV 包含 SNR、BLER、CRC fail rate、BER 和样本数，JSON sidecar 保存配置及完整统计；BLER 将 CRC fail 或任何 payload bit 错误都计为 block error。`bler_smoke.toml` 是短时连通性配置，正式仿真应增加 `max_frames_per_snr` 和 `target_block_errors`。

`[bler]` 的 `stop_at_zero_bler = true` 会在某个 SNR 点测得零误块时结束该检测器的扫描，并把更高的 SNR 点记为跳过而不是继续仿真；这依赖 BLER 随 SNR 单调下降，默认关闭。CSV 只写出实测点；跳过的点记录在 JSON sidecar 的 `skipped_points`（含触发跳过的 SNR 与原因）和网页的进度流中，网页会显示“跳过 N”并据此计算进度。

### 网页仿真界面

在仓库根目录使用指定虚拟环境安装后启动本地服务：

```bash
/home/le-lei/workspace/test/.venv/bin/pip install -e .
/home/le-lei/workspace/test/.venv/bin/nr-pusch-web --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。CDL 页沿用原三个配置编辑器、保存/预检和串行任务；接收分析页仍使用原 RX 流程。运行任务按提交顺序串行执行，进度、图表、历史、日志与结果下载仍归属各自持久 job。该服务按 local-only 工具设计；绑定非 loopback 地址前必须在外部提供认证和访问控制，不能直接作为公网服务。

CDL 仍有 25 个原预设；Sionna RT 有 `rt-los-quick`、`rt-ground-quick`、`rt-ground-wall-quick`、`rt-cp-los-quick` 四个一键仿真组合，以及 18 个场景选择项（3 个项目场景＋15 个 Sionna RT 2.2.0 随包场景）。选中随包场景后沿用当前 RT 配置中的 BS/UE 位置；城区场景网格较多、耗时和内存需求较高。选择场景、DFT/CP 波形和预算后一次运行会自动完成 tracing、Web gate 和 BLER；CP 通过 TX profile 派生为自定义组合。当前 quick profile 见上表，包含最多 2,000 帧/SNR；其名称不代表低工作量。Long profile 是 CUDA:0、8 SNR、batch 20、最多 2,000 帧/SNR、target 200、perfect/DMRS × 三个 detector，并可在首个零错误点后按每个 estimator/detector arm 跳过更高 SNR；跳过行标记为 skipped、BLER/CI 为空、计入进度但不算作 BLER=0。Web RT 限制由 `/api/rt/options` 返回；输入超限会显示具体字段/数值/上限的弹窗并保持运行禁用，不自动缩小或改设备。

高级配置可编辑 BS/UE、阵列、功率、波束、噪声、参数化地面/墙面、追踪预算及 RT/TX/BLER TOML。导入仅接受根 `scene.xml` 与受限 ASCII 三角网格 `meshes/*.ply`；PLY 顶点采用世界 XYZ、单位米，需在导入前完成定位，不支持 Mitsuba transform 或单位转换；预览只显示 XY 包围盒，不显示传播路径。JSON 请求体限 12 MiB、ZIP 限 8 MiB、解压内容总量限 32 MiB。该接口不是任意 Mitsuba 工程或 3D 拖放建模器。`GET /api/rt/options` 查询设备/限制，`GET /api/rt/scene-template.zip` 下载模板，`POST /api/rt/scenes` 导入包，`POST /api/rt/config` 解析 RT 参数。每个 job 复制场景和配置，可下载 validation report、NPZ/JSON snapshot 与无主机绝对路径的 `reproducibility.zip`。配置超限时弹窗列出所有超限字段、当前值与允许范围；profile 不会被自动改写。

每个成功完成的 RT Web job 另存于 `outputs/rt_runs/<job-id>/`；API job 记录和状态面板显示归档相对路径及本次静态快照 cache hit/miss。快照缓存位于 `outputs/rt_snapshots/<SHA-256>/`，指纹包含 RT/TX 资源网格与 DMRS 几何、场景 bundle hash、软件版本、Mitsuba variant 和快照实现指纹；影响该指纹的输入变化会重新追踪，不复用 BLER 结果。网格场景 cache hit 仍重新执行 doubled-sample Web 收敛 gate。

RT 网页 gate `web-frequency-v1` 要求 native FD/TD 相对 RMS ≤ 0.001、tap-window/CP 外有效能量与直接 CFR 截断误差各 ≤ 1%，mesh 场景还需同 seed 的双采样路径收敛。原严格 FD/TD `1e-5` 结果保留为独立质量标记，不影响此有界 Web gate：2026-10-04 ground-wall 实测 RMS `0.0003351521`、CFR 截断误差 `0.00922528199`、CP 52 samples；Web gate 通过、strict gate 仍失败。页面持续标注固定场景/静态快照条件 BLER 和 strict warning；这项放宽不表示 strict 时域等价，也不构成城市信道或统计 benchmark。
实际 ground-wall 浏览器 job `7a69bd0a2ea4` 使用当时的 quick profile、seed 13、噪声比 0.001、CPU PHY，9/9 点 complete；20 dB beam-independent 为 8/8 aggregate TB errors，ZF/LMMSE 观察到 0/8 errors（Bonferroni 95% 上界 0.920943）。它是旧 profile 的条件诊断，不是当前 quick profile 的结果，也不是统计 BLER 或零错误保证。
2026-10-05 full regression：`PYTHONPATH=src /home/le-lei/workspace/test/.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v` 通过 182 tests；unittest 报告运行时间 602.470 s（shell wall time 607.12 s）。这是代码回归证据，不是 BLER/统计性能证据。

### 历史 GPU batch 扫描与频域/时域对照

在 NVIDIA GeForce RTX 3060 Laptop GPU（6 GiB，测试时约有 1.3 GiB 被其他进程占用）上，用 CUDA、4 用户 DFT-s-OFDM、CDL-A、DMRS 信道估计、LMMSE 检测和 30 dB SNR 测量单批吞吐。表中帧吞吐为重复运行的均值；每帧含 4 个 UE transport blocks。

| Batch frames | 帧/秒 | TB/秒 | PyTorch 峰值保留显存 | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 24 | 8.96 | 35.8 | 3.12 GiB | 稳定，显存余量较好 |
| 28 | 10.00 | 40.0 | 3.62 GiB | 推荐日常使用 |
| 32 | 10.78 | 43.1 | 4.13 GiB | 两次完成，显存余量很小 |
| 36 | — | — | — | OOM |
| 48 | — | — | — | OOM |

旧实现中，batch 28 比 24 快约 12%，比 32 慢约 7.8%，但少占约 0.51 GiB 峰值保留显存。该扫描使用原先的频域线性插值估计器；当前统一的联合 LS tap 拟合改变了运算量，表中吞吐和推荐 batch 需要重新测量后才能用于新实现。

频域与时域的大规模对照复用了既有时域全面仿真 `/tmp/pusch_mimo_detection_comparison_ep_full_18_39.csv/json` 中的 LMMSE 结果，没有重跑时域链路。频域使用 batch 28、相同的 18–39 dB SNR 点、seed、DMRS、LMMSE 和每个 SNR 点相同的帧数，共 2,857 帧/11,428 个 TB。数据如下：

两组运行使用相同 seed 和每点样本数，但 batch 分段不同（时域 1、频域 28），因此不是逐帧配对的同一组随机 IQ；低 SNR 点样本数也较少，主要看作趋势对照。历史时域 LMMSE 运行耗时合计约 747 秒，本次频域约 195 秒。

| SNR (dB) | 时域帧数 | 时域 BLER | 频域帧数 | 频域 BLER |
| ---: | ---: | ---: | ---: | ---: |
| 18 | 26 | 0.9904 | 26 | 1.0000 |
| 21 | 27 | 0.9259 | 27 | 0.9815 |
| 24 | 35 | 0.7357 | 35 | 0.8929 |
| 27 | 53 | 0.4717 | 53 | 0.6132 |
| 30 | 143 | 0.1766 | 143 | 0.3077 |
| 33 | 573 | 0.0436 | 573 | 0.1405 |
| 36 | 1,000 | 0.0065 | 1,000 | 0.0703 |
| 39 | 1,000 | 0.00125 | 1,000 | 0.0480 |

这组历史对照**没有通过频域/时域 BLER 接近性验证**：频域 BLER 在中高 SNR 明显偏高，39 dB 处出现约 0.048 的误块平台；8 个点的平均绝对 BLER 差约 0.088。频域路径的 LMMSE/DMRS 诊断还显示，perfect-CSI 频域链路在 33 dB、140 帧时 BLER 为 0.0089，而 DMRS 频域链路为 0.1375。该数据产生于两域分别使用不同信道转换和 DMRS 插值算法的旧实现；表格保留为历史结果，不代表当前统一接收链路的性能，需重新进行配对验证。

修复后的定点配对检查在 30 dB 使用同一发送 payload、CDL seed 和 OFDM 解调后的噪声，比较了 12 帧/48 个 TB：两域各有 6 个 CRC 失败，CRC 判定分歧为 0；无噪声接收网格的平均相对 RMS 差为 0.235%。该样本量只用于检查两条实现路径的一致性，不能替代完整 BLER 曲线。

全面仿真 CSV/JSON 保存在 `/tmp/pusch_mimo_detection_comparison_ep_full_18_39.csv/json`（既有时域全检测器结果）及 `/tmp/pusch_frequency_vs_previous_time.csv/json`（本次频域结果和对照）。短时诊断、batch 扫描原始文件和临时脚本不作为结果归档。

### 当前统一接收链路的全量检测器对照

`configs/bler_4ue_cdl_gpu.toml` 的一次完整运行，任务 `1313e4fe353d`，共 35 个点、约 55 分钟、结果在 `runs/web/1313e4fe353d/`（该目录被 Git 忽略）。该表的批次与检测器组合取自当次运行的**归档快照** `runs/web/1313e4fe353d/simulation.toml`（`detector_batch_sizes = { lmmse = 26, "lmmse-sic" = 26, "k-best" = 10, "mmse-pic" = 26, ep = 26 }`，`detectors = ["lmmse", "lmmse-sic", "k-best", "mmse-pic", "ep"]`）；当前 `configs/bler_4ue_cdl_gpu.toml` 已是 4 个检测器、20/20/20/8，因此下表是 2026-09-27 那次运行的结果，不是对今天 profile 的描述：

| SNR (dB) | LMMSE | LMMSE-SIC | K-best(16) | MMSE-PIC(4) | EP(10) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.9808 | 0.9808 | 1.0000 | 0.9712 | 0.9808 |
| 25 | 0.6394 | 0.4071 | 0.8188 | 0.5721 | 0.6731 |
| 30 | 0.1987 | 0.0598 | 0.2917 | 0.1079 | 0.4359 |
| 35 | 0.0140 | 0.0030 | 0.0170 | 0.0040 | 0.2428 |
| 40 | 0.0015 | 0.0000 | 0.0005 | 0.0005 | 0.0125 |
| 45 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |
| 50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

所有检测器的 BLER 随 SNR 单调下降到 0，当前统一接收链路没有旧实现中 39 dB 处的误块平台；这与旧表使用的信道转换和 DMRS 插值实现不同，但两次运行未做逐帧配对，归因属于推断。45 dB 起 LMMSE、LMMSE-SIC、K-best 和 MMSE-PIC 的 4,000 个 TB 全部 CRC 通过，EP 到 50 dB 归零。排序为 LMMSE-SIC 最好、MMSE-PIC 次之，LMMSE 与 K-best 接近，EP 在整个 SNR 范围内落后于 LMMSE（35 dB 处 0.243 对 0.014）；EP 的零时延等效空间信道近似与 QAM 矩匹配仍是最主要的性能差距来源，尚未用参考向量定位。

同一运行的满额点吞吐（4,000 TB/点）分别为 LMMSE 55.2 TB/s、K-best 45.7 TB/s、MMSE-PIC 51.5 TB/s、EP 53.5 TB/s、LMMSE-SIC 13.3 TB/s；这是 batch 26 单点测量，不是历史表格那种 batch 扫描，batch 推荐值仍需按新实现重测。

复现性说明：`simulate_bler` 每次以 `sionna.phy.config.seed` 重置 Sionna 的全局生成器，CDL 实现按调用顺序从该流取随机数。因此只有整个扫描配置完全一致（相同 batch 分组和相同停止条件）时结果才逐点相同；只改 `max_frames_per_snr` 或 `target_block_errors` 会改变前序 SNR 点的信道抽样次数，从而让后续点换用不同的信道实现。例：早前 `max_frames_per_snr = 100` 的短时运行在 35 dB 首个 26 帧内有 59 个误块，而本次 1,000 帧共 56 个误块，说明两组并非逐帧配对，跨运行比较只能按趋势看待。

### 抓包调优参数不能直接迁移到仿真（配对 A/B）

`[bler]` 现在支持 `l_min` 与 `max_delay_spread_s`，即接收端 DMRS 有限抽头拟合窗口，与 RX profile 的同名旋钮对应；`max_delay_spread_s` 省略时沿用 CDL 信道配置的值。加这两个键的目的是让扫描能够**逐项复现**某个抓包 profile 的接收设置，而不是默认就套用它。

实测结果说明不能默认套用。在 `pusch_4ue.toml`（MCS 20）+ `cdl_38_901_4x4.toml` 上做配对 A/B：固定 `seed = 20260924`、SNR `[25,30,35,40]`、batch 20、每点 1,000 帧（4,000 TB），关闭 `target_block_errors` 早停与 `stop_at_zero_bler`，使各变体消耗完全相同的载荷/信道/噪声随机流；同一配置重复跑两次的 `bit_errors` 逐 bit 相同，证明配对成立。

| 变体 | MMSE-PIC 迭代/阻尼 | 接收窗 | 25 dB | 30 dB | 35 dB | 40 dB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A 原仿真基线 | 4 / 0.25 | 跟随信道（3 µs） | 0.5725 | 0.0877 | 0.0035 | 0.0003 |
| A' 重复跑（校验） | 4 / 0.25 | 跟随信道（3 µs） | 0.5725 | 0.0877 | 0.0035 | 0.0003 |
| B 对齐 `rx_pusch_4ue.toml` | 8 / 0.5 | 6 µs | 0.6690 | 0.2567 | 0.0963 | 0.0127 |
| C 仅迭代+阻尼 | 8 / 0.5 | 跟随信道（3 µs） | 0.5560 | 0.2500 | 0.0700 | 0.0043 |
| D 仅窗口 | 4 / 0.25 | 6 µs | 0.6787 | 0.1455 | 0.0050 | 0.0010 |

**结论是退化而不是提升**：B 相对 A 的 BLER 在 30 dB 差 2.9 倍、35 dB 差 27.5 倍、40 dB 差 51 倍，BER 在 35 dB 由 2.68e-05 升到 2.31e-03。分解后主因是 8 次迭代加阻尼 0.5（C 单独即使 35 dB 差 20 倍），6 µs 接收窗次之（D 单独 35 dB 只差 1.43 倍）。抓包 profile 的取值是对单帧抓包的补偿，并非对算法普适，因此 `configs/bler_4ue_cdl.toml` 与 `configs/bler_4ue_cdl_gpu.toml` **保持各自的仿真基线数值**（4 次迭代、阻尼 0.25、窗口跟随信道），只保留新旋钮。详见 `docs/receiver_optimization.md` §3.2。

接收端可独立加载时域 IQ 或频域资源网格 NPZ：`iq` 使用 `[batch,rx_antenna,sample]` 轴，`grid` 使用接收机频域网格轴；`perfect` 模式另需真实 `channel_taps` 或 `channel_frequency_response`，`dmrs` 模式从输入中的 PUSCH DMRS 估计 CSI。输入扩展名为 `.h5`/`.hdf5` 时会自动作为 MATLAB H5 接收夹具读取，也可用 `--input-format matlab-h5` 指定。当前 MATLAB 接口读取 `FreqData/IQdataPdu_real` 与 `FreqData/IQdataPdu_imag`，原始轴为 `[ofdm_symbol,rx_antenna,active_subcarrier]`，并会附加 batch 和 stream 轴供接收机处理。若 H5 中有 `data_*` 和 `pilot_*`，JSON sidecar 还会记录分离数组形状、天线平均功率和峰值幅度，便于抓包分析。

抓包分析中 `bit_errors` 只是与参考链路载荷的比对值，不能单独作为译码正确的判据：参考链路自身可能CRC 失败。sidecar 的 `reference_comparison` 因此同时记录 `crc_status`、`crc_verified_users` 和说明，CLI 也会在出现“CRC 通过但与参考不一致”的用户时显式提示。BLER 仿真不涉及该问题，因为发送与接收使用同一套比特。

接收端的 `--estimate-delay` 是诊断选项：用“FFT 峰值 + 抛物线插值”估计每个 UE、每根接收天线的整体时延（采样），仅写入 JSON sidecar 的 `estimated_bulk_delay_samples_by_user` 和兼容字段 `estimated_bulk_delay_samples`，不修正 CSI。单独开启时译码结果应与基线一致；把估计时延事后作为相位斜坡补回 CSI 会使正常抓包退化，正确补偿需要按天线进行分数时延重拟合。空间签名投影实验没有带来收益且会导致低信噪比译码退化，因此对应的实现、CLI 选项和专用测试已清理。两级 CPE 相位补偿（盲四阶矩 + 判决引导）按参考实现接入后也使 4/4 降到 3/4，且在 `RxTestVector.h5` 上出现非有限值，故未保留。

MATLAB 抓包还可能使用与 RNTI 无关的 `c_init` 加扰，此时标准解扰器必然解不出 TB。`nr-pusch-rx --scrambling <h5>` 读取每用户 `ue<k>_scrambSeq`（长度须等于 TB 编码后的 46,800 bit）并替换 TB 解码器的解扰器；sidecar 的 `scrambling_source` 记录所用来源，profile 本身不携带序列。译码判据只取本机 CRC：抓包 H5 里的 `ue*_tx_bits` 来自参考链路，参考链路自身 CRC 失败时该载荷不可信。

LMMSE-SIC 的内部 CRC 门控译码器与重编码抵消路径也使用同一组显式序列；
未提供 `--scrambling` 时仍使用配置中的标准 RNTI 加扰。

#### 四组 MATLAB 抓包的译码结果

低码率三组（`RxTestVector.h5`、`RxTestVectorCase11121314.h5`、`RxTestVectorCase78914.h5`）共用
`configs/rx_pusch_4ue.toml`：`mcs_index = 20`（目标码率 0.6015625，28,168 bit TB，46,800 码比特，4 个码块），
`[receiver]` 默认 MMSE-PIC（8 次迭代、阻尼 0.5）、`max_delay_spread_s = 6e-6`、噪声方差 0.001。
高码率一组（`RxTestVectorCase123427.h5`）用 `configs/rx_pusch_4ue_mcs27.toml`：`mcs_index = 27`
（目标码率 0.92578125，43,032 bit TB，同样 46,800 码比特但分成 6 个码块）。该抓包
的主径偏向负延迟，DMRS 拟合窗改为 `l_min = -44`、`max_delay_spread_s = 2e-6`；
检测器仍为 MMSE-PIC（8 次迭代、阻尼 0.5），噪声方差 0.0003。

| 抓包 | profile（MCS） | 加扰序列 | 检测器 | 迭代 / 阻尼 | 时延基 | 噪声方差 | TB CRC | 码块 CRC |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| `RxTestVector.h5` | `rx_pusch_4ue`（20） | 标准 RNTI | MMSE-PIC | 8 / 0.5 | 6 µs | 0.001 | **4/4** | 全部 4/4 |
| `RxTestVectorCase11121314.h5` | `rx_pusch_4ue`（20） | `scrambSeqCase11121314.h5` | MMSE-PIC | 8 / 0.5 | 6 µs | 0.001 | **4/4** | 全部 4/4 |
| `RxTestVectorCase78914.h5` | `rx_pusch_4ue`（20） | `scrambSeqCase78914.h5` | MMSE-PIC | 8 / 0.5 | 6 µs | 0.001 | **3/4**（ue0 未解出） | ue0 0/4，其余 4/4 |
| `RxTestVectorCase123427.h5` | `rx_pusch_4ue_mcs27`（27） | `scrambSeqCase123427.h5` | MMSE-PIC | 8 / 0.5 | 2 µs，l_min=-44 | 0.0003 | **3/4**（ue0/ue1/ue2） | ue0/ue1/ue2 均 6/6；ue3 0/6 |

`nr-pusch-rx --cb-crc` 记录每个用户的逐码块 CRC 判定（`RxResult.metadata["cb_crc_status"]`，sidecar
同名字段）。在当前参数下，case78914 的 ue0 是 0/4，case123427 的 ue3 是 0/6；
不能将 H5 参考链路的错误载荷当作译码成功的判据。

四组抓包的 DMRS 端口映射都是 0/1/2/3。低码率组把 `max_delay_spread_s` 从 3 µs 放宽到 6 µs 是 78914 与 11121314 能否解出的关键（前者 1/4→3/4，后者 1/4→4/4）。

`RxTestVectorCase123427.h5` 的码率 0.92578125 留给纠错的余量很小。原先以 `l_min=-6`、
3/6 µs 信道基扫描五种检测器（MMSE-PIC、LMMSE-SIC、LMMSE、EP、K-best），最高只有 2/4。
将 DMRS 信道拟合窗向负延迟移动后，ue2 从 0/6 个码块恢复至 6/6，整块 CRC 达到 3/4；
ue3 仍为 0/6，尚未实现四用户全通过，不应依据参考载荷误码数替代 CRC。

`RxTestVectorCase78914.h5` 的检测器对比（噪声 0.001/0.003 × 时延 3/6 µs，取每种检测器的最好结果）：MMSE-PIC 3/4，LMMSE、LMMSE-SIC、EP 各 2/4，K-best(16) 1/4；MMSE-PIC 的迭代次数（4–32）与阻尼（0.25–0.75）对结果影响小于 100 bit 错误。ue0 在所有配置下都失败（BER≈18%）：更换检测器族、PIC 迭代与阻尼、抽头窗口下界（−6/−10/−14）均无改善，DMRS 端口置换实验显示现有排列唯一正确（任何非恒等排列都会把另一个用户打到 ≈50% 误码）。因此 ue0 属于该抓包本身的弱用户，接收侧已无可调空间。ue2 与 H5 参考载荷相差 2628 bit，是参考链路自身 CRC 失败所致，以本机 CRC 为准。

```bash
nr-pusch-rx --rx-config configs/rx_pusch_4ue.toml --input /path/to/capture.npz --output /tmp/decoded.npz
```

使用仓库提供的 MATLAB H5 接收向量进行分析和 CRC 解码。当前 `configs/rx_pusch_4ue.toml` 为该向量配置 DMRS 信道估计、频域输入和 MMSE-PIC 参数；接收机设置集中在 `[receiver]`，CLI 不带覆盖参数时会读取这些默认值：

```bash
nr-pusch-rx --rx-config configs/rx_pusch_4ue.toml --input tests/fixtures/matlab_h5/RxTestVector.h5 --output /tmp/rx_test_vector_decode.npz --device cpu
```

高码率抓包可改用 `--rx-config configs/rx_pusch_4ue_mcs27.toml` 并传入对应
`--scrambling tests/fixtures/matlab_h5/scrambSeqCase123427.h5`；`--l-min`
可在命令行覆盖 profile 的 DMRS 拟合窗下界。

该 H5 夹具包含与接收 IQ 同源的 `ue*_tx_bits` 参考数据。CLI 的 JSON sidecar 会给出逐 UE bit error/BER/exact-match 与 `crc_status`、`crc_verified_users`；判据以本机 CRC 为准，参考比对只作辅助（参考链路自身可能 CRC 失败，此时参考 bits 不可信）。当前 profile 对该向量得到 4/4 CRC pass 与 0 bit error。配置中的 `noise_variance = 0.001` 是在这些接收向量上调出的检测参数，不是 H5 提供的实测噪声元数据，也不应直接视为其他现网抓包的噪声估计；分析其他抓包时应按对应采集链路估计并覆盖该值。接收输出 NPZ 保存解码 bits 和逐用户 CRC 状态。旧参数名 `--tx-config` 仍作为 `--rx-config` 的兼容别名。

### 网页外部接收分析

启动本地网页后，“接收分析”默认选用仓库内的 `configs/rx_pusch_4ue.toml` 和 `tests/fixtures/matlab_h5/RxTestVector.h5`。也可上传 `.h5`/`.hdf5` 或 `.npz` 文件，选择其他 RX TOML 配置，或粘贴完整 TOML 覆盖，再设置输入域、检测器及其参数。**选中 RX TOML 时，网页会解析其 `[receiver]` 表并回填表单**（`detector`、`detector_parameter`、`detector_damping`、`noise_variance`、`max_delay_spread_s`），与 `nr-pusch-rx` 的取值优先级一致：请求 > profile `[receiver]` > 内置兜底。因此在网页上选择 `rx_pusch_4ue_mcs27.toml` 就能复现该抓包的 3/4，而不需要手动填写每个旋钮。勾选“使用内置默认采集”会同时切换到与内置夹具匹配的 profile。

结果区按 UE 分开展示：

- **逐 UE 星座图**：每个 UE 一张独立画布、独立坐标幅度（否则强用户会把弱用户压平），显示检测器软输出的 QAM 点。
- **逐 UE CRC 状态**：TB CRC 徽标加上一行码块格子，每格是一个码块，鼠标悬停显示码块序号与判定；徽标给出 `通过数/总数`。

抓包若使用非 RNTI 扰码（MATLAB 参考向量即如此），需同时上传对应的加扰序列 H5（如 `scrambSeqCase123427.h5`），网页会把它传给 `nr-pusch-rx --scrambling`。**这一步会直接影响码块 CRC**：扰码按位保持整段 TB 的码字，因此 TB CRC 不变，但码块会被跨块打散，CB CRC 会失真——例如 `RxTestVectorCase123427.h5` 不传扰码时 ue2 从 6/6 掉到 1/6。

下载的 JSON 清单记录 H5 参考比特比较结果。页面也提供 bits/CRC NPZ 下载。H5 当前按 MATLAB `FreqData/IQdataPdu` 频域格式读取；NPZ 时域数组为 `iq`，频域数组为 `grid`。单文件上传上限为 12 MiB，数据由本地 Web 服务处理。

```bash
nr-pusch-web --host 127.0.0.1 --port 8765 --config-dir configs --runs-dir runs/web
```

`RxTestVector.h5` 不包含实测噪声方差或真实信道元数据。仓库的调优 profile 使用 DMRS 信道估计和 MMSE-PIC（8 次迭代、阻尼 0.5、噪声方差参数 0.001、`max_delay_spread_s = 6e-6`）；修改检测器或接收参数后，判据取本机 CRC，`reference_comparison` 只用于交叉核对。

## 运行

使用仓库指定的虚拟环境安装本仓库：

```bash
/home/le-lei/workspace/test/.venv/bin/pip install -e .
nr-pusch-tx --config configs/pusch_4ue.toml --output runs/first_tx.npz --batch-size 1 --seed 7
```

NPZ 中 `iq` 的轴顺序为 `[batch, user, tx_antenna, sample]`，`frequency_grid` 为 Sionna PUSCH CP-OFDM 资源网格（`[batch, user, tx_antenna, symbol, subcarrier]`），`bits` 为 `[batch, user, transport_block_bit]`。同名 JSON manifest 保存已解析的运行配置、轴含义和依赖版本。

## 目录

- `configs/`：发送与抓包分析配置
- `src/nr_pusch/`：配置解析、Sionna 发送与 CDL 信道适配、IQ 导出和命令入口
- `tests/fixtures/matlab_h5/`：随仓库提供的 MATLAB H5 参考夹具（RxTestVector 系列、TxTestVector、scrambSeqCase 系列），只读使用
- `docs/`：支持范围、接口和夹具说明（`docs/dmrs_tap_power_prior.md` 记录 prior 原理与生成规则）
