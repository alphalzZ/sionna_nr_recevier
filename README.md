# NR PUSCH Lab

一个小型、配置驱动的 5G NR PUSCH 发送与 IQ 分析仓库。首批目标为 4 用户上行 MU-MIMO，每用户 1 层；发送端通过独立 IQ 文件和运行清单与接收端解耦。

## 当前发送端

当前实现以 Sionna 2.0.1 的 `PUSCHTransmitter` 生成 PUSCH 数据资源网格，并支持 CP-OFDM 和 DFT-s-OFDM 波形输出。DFT-s-OFDM 对数据符号应用酉 DFT，并按 type-1 低 PAPR 序列生成 DMRS 后进行 OFDM 调制。由于 Sionna 2.0.1 的组合 PUSCH 发射器不支持 transform precoding，配置中的 DFT 预编码 MCS 会映射到同调制阶数和码率的原生 Sionna MCS，以复用其 TB 编码器。当前配置为 DFT-s-OFDM MCS table 1/index 20，目标码率 0.6015625，TB 大小 28,168 bit。

## CDL 信道

`NrPuschCdlChannel` 使用 Sionna TR 38.901 CDL A–E 模型处理 4 个独立的上行 UE 链路，每个 UE 使用 1 根发射天线；4 条链路在 4 天线 BS 接收端叠加。接收阵列形状、CDL 模型、载波频率、时延扩展和速度范围等参数位于 `configs/cdl_38_901_4x4.toml`。该 profile 使用单极化 2×2 接收阵列。

BLER 链路默认用 `channel_domain = "frequency"`：按 OFDM 符号采样 CDL CIR，使用与时域链路相同的有限长度 sinc taps 和时延范围生成频响，再直接施加到发射资源网格，保留 batch 并行，不生成逐采样 CIR 和时域卷积。该单抽头频域模型假设 CP 足以覆盖时延扩展，不模拟 CP 不足导致的 ISI 或符号内快速时变造成的 ICI。设为 `channel_domain = "time"` 可使用时域卷积；独立 IQ 抓包的信道命令也继续使用时域路径。

先生成发送 IQ，再将其作为独立输入应用信道：

```bash
nr-pusch-tx --config configs/pusch_4ue.toml --output /tmp/pusch_tx.npz --seed 7
nr-pusch-channel --config configs/cdl_38_901_4x4.toml --input /tmp/pusch_tx.npz --sample-rate-hz 18000000 --output /tmp/pusch_rx.npz
```

发送输入必须为 NPZ 中的 `iq` 数组，轴顺序为 `[batch,user,tx_antenna,sample]`。信道结果 `iq` 轴顺序为 `[batch,rx_antenna,sample]`，`per_user_iq` 为 `[batch,user,rx_antenna,sample]`。输出包含线性卷积产生的信道尾部；未加入噪声和路径损耗，默认保留 CDL 的实际归一化功率设置。

离线资源网格也可直接应用频域信道；输入使用发送 NPZ 中的 `frequency_grid`，并提供发送 TOML 以取得 FFT/子载波间隔配置：

```bash
nr-pusch-channel --config configs/cdl_38_901_4x4.toml --input /tmp/pusch_tx.npz --sample-rate-hz 18000000 --domain frequency --tx-config configs/pusch_4ue.toml --output /tmp/pusch_rx_grid.npz
nr-pusch-rx --tx-config configs/pusch_4ue.toml --input /tmp/pusch_rx_grid.npz --input-domain frequency --channel-estimator perfect --noise-variance 0.001 --output /tmp/pusch_decode.npz
```

频域信道 NPZ 提供 `grid`、`per_user_grid` 和 `channel_frequency_response`，可直接输入接收机，不进行 OFDM 调制、时域信道卷积或接收端 OFDM 解调。

## 多用户接收和 BLER

`NrPuschRx` 使用 Sionna PUSCH TB 解码器、4×4 MIMO 检测，以及 DFT-s-OFDM 每数据符号的逆 DFT。时域 IQ 先由 Sionna OFDM 解调为资源网格；直接加载的频域网格跳过这一步。两种输入随后共用同一个频域 DMRS 估计器：利用发送端实际映射的低 PAPR DMRS 和 OCC 端口序列，对每对共享 comb 的用户做联合 LS 拟合，输出各子载波频响和估计误差方差。其流程遵循 [Sionna OFDM MIMO 信道估计与检测教程](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/OFDM_MIMO_Detection.html) 的资源网格导频估计与检测接口；Sionna 原生 PUSCH 导频序列与本仓库的 DFT-s-OFDM 序列不同，因此此处保留自定义的 OCC 处理。`perfect` 模式直接使用仿真 CDL CSI，仅用于上界对照。

DMRS 的 LS 拟合使用配置的 `max_delay_spread_s` 限制候选 tap 范围：当前为 18 MHz 采样率下的 `-6..60`，共 67 taps。这是接收机的时延范围先验，不读取本帧的真实信道系数。单个 DMRS 符号的估计扩展到整个 slot，适用于当前零速静态 CDL profile；高 Doppler 或多 DMRS 配置需扩展时频插值并用抓包参考验证。

检测器可选 `lmmse`、`lmmse-sic`、`k-best`、`mmse-pic` 和 `ep`，参数通过 `detector`、`detector_parameter` 或 BLER 配置中的 `detector_parameters` 配置，接收 CLI 也提供 `--detector` 和 `--detector-parameter`。`k-best` 和 `ep` 均先做频域 LMMSE 预均衡并 IDFT；每个时域采样点建立四流空间模型，频率变化与等化噪声合并为残余 ISI 协方差。K-best 在该模型上搜索有限星座路径。EP 则为四个 QAM 用户维护复高斯近似因子，以 cavity 分布对离散星座做矩匹配，并对因子参数阻尼迭代，最后由后验均值和方差形成软 LLR。`mmse-pic` 从时域 LLR 计算软星座期望，DFT 回频域后并行消除其他 UE，并更新 LLR。迭代次数通过 `detector_parameters` 设置；软迭代阻尼可用 `detector_damping` / `--detector-damping` 配置。`lmmse-sic` 按估计信道功率从强到弱处理 UE：仅在该 UE CRC 通过时重编码、重构其 DFT-s-OFDM 资源网格并消除干扰。单次接收 JSON sidecar 会记录 SIC 检测顺序和每个 UE 消除前的 CRC 状态。BLER 配置中的 `detectors` 会按相同 seed、CDL、SNR 和 payload 顺序比较检测器；例如 `{ "k-best" = 16, "mmse-pic" = 4, "ep" = 10 }`。

EP 与 K-best 共用零时延等效空间信道近似；滤波后剩余的频率选择性记入高斯协方差，而非在 EP 图中显式建模所有跨采样相关性。该近似、软 LLR 校准及收敛行为仍需通过 MATLAB 参考向量和长 SNR 曲线验证。

配置 SNR 扫描点、批大小、帧上限和停止错误数后运行：

```bash
nr-pusch-bler --tx-config configs/pusch_4ue.toml --channel-config configs/cdl_38_901_4x4.toml --simulation-config configs/bler_smoke.toml --output /tmp/pusch_bler.csv --device cpu
```

将 `device` 写为 `cpu`、`cuda:0` 或 `auto` 可从 TOML 控制运算设备；`configs/bler_4ue_cdl_gpu.toml` 提供显式 CUDA 示例。若以 `--device cuda` 覆盖，程序会将其规范化为 Sionna 接受的 `cuda:0`。请求的设备同时会写入 Sionna 的全局 `sionna.phy.config.device`：Sionna 的 PUSCH 导频图案按该全局值分配，而资源网格按显式设备分配，两者不一致时（例如在有 GPU 的机器上跑 `--device cpu`）会在建图阶段报设备不匹配。批大小受 GPU 显存约束；频域信道一般比时域线性卷积更节省显存。

不同检测器可在 `[bler]` 中单独设置批大小；未列出的检测器使用 `batch_size`，每个 SNR 点最后一批仍受 `max_frames_per_snr` 截断。例如 `detector_batch_sizes = { lmmse = 20, "lmmse-sic" = 20, "k-best" = 1, "mmse-pic" = 1, ep = 1 }`。此前两次本机 GPU 运行均在 batch 20 下完成 LMMSE 和 LMMSE-SIC，随后在 K-best 阶段 CUDA OOM；示例因此将 K-best 改为 1。MMSE-PIC 和 EP 的值也是保守起点，尚未测定最优吞吐，可逐项调高。网页的仿真 TOML 编辑器同样支持该字段。不同批大小改变随机数的分组，因此各检测器共享初始 seed 和统计条件，但不保证逐帧使用完全相同的 payload、信道和噪声样本。

扫描会逐 SNR 点发射随机 transport blocks、通过 CDL、按每个接收天线的测得信号功率注入复 AWGN，再用 CRC 与 payload 比对统计 BLER。CSV 包含 SNR、BLER、CRC fail rate、BER 和样本数，JSON sidecar 保存配置及完整统计；BLER 将 CRC fail 或任何 payload bit 错误都计为 block error。`bler_smoke.toml` 是短时连通性配置，正式仿真应增加 `max_frames_per_snr` 和 `target_block_errors`。

`[bler]` 的 `stop_at_zero_bler = true` 会在某个 SNR 点测得零误块时结束该检测器的扫描，并把更高的 SNR 点记为跳过而不是继续仿真；这依赖 BLER 随 SNR 单调下降，默认关闭。CSV 只写出实测点；跳过的点记录在 JSON sidecar 的 `skipped_points`（含触发跳过的 SNR 与原因）和网页的进度流中，网页会显示“跳过 N”并据此计算进度。

### 网页仿真界面

在仓库根目录使用指定虚拟环境安装后启动本地服务：

```bash
/home/le-lei/workspace/test/.venv/bin/pip install -e .
/home/le-lei/workspace/test/.venv/bin/nr-pusch-web --host 127.0.0.1 --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。页面可选择并编辑发送、CDL 信道和 BLER TOML profile；“保存配置”会校验并覆盖所选 `configs/*.toml`，而直接“启动仿真”使用当前编辑器文本创建独立快照，无需先保存。建议先用 `bler_smoke.toml` 熟悉操作，再选择完整扫描配置。运行任务按提交顺序串行执行，避免多个任务同时争用 GPU；页面显示每个检测器和 SNR 点的进度、BLER 曲线、统计表和日志，结果可下载为 CSV/JSON。任务及配置快照保存在被 Git 忽略的 `runs/web/<任务 ID>/`，服务重启后仍可查看已有结果；正在运行的进程因服务中断而结束时，任务会标记为失败。网页只绑定本机回环地址；需要从其他机器访问时请使用 SSH 端口转发。

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

`configs/bler_4ue_cdl_gpu.toml`（SNR 20–50 dB、`max_frames_per_snr = 1000`、`target_block_errors = 100`、seed 20260924、频域信道、DMRS 估计、每检测器 batch 26/26/10/26/26）的一次完整运行，任务 `1313e4fe353d`，共 35 个点、约 55 分钟、结果在 `runs/web/1313e4fe353d/`（该目录被 Git 忽略）：

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

接收端可独立加载时域 IQ 或频域资源网格 NPZ：`iq` 使用 `[batch,rx_antenna,sample]` 轴，`grid` 使用接收机频域网格轴；`perfect` 模式另需真实 `channel_taps` 或 `channel_frequency_response`，`dmrs` 模式从输入中的 PUSCH DMRS 估计 CSI。输入扩展名为 `.h5`/`.hdf5` 时会自动作为 MATLAB H5 接收夹具读取，也可用 `--input-format matlab-h5` 指定。当前 MATLAB 接口读取 `FreqData/IQdataPdu_real` 与 `FreqData/IQdataPdu_imag`，原始轴为 `[ofdm_symbol,rx_antenna,active_subcarrier]`，并会附加 batch 和 stream 轴供接收机处理。若 H5 中有 `data_*` 和 `pilot_*`，JSON sidecar 还会记录分离数组形状、天线平均功率和峰值幅度，便于抓包分析。

```bash
nr-pusch-rx --rx-config configs/rx_pusch_4ue.toml --input /path/to/capture.npz --noise-variance 0.001 --channel-estimator dmrs --detector lmmse-sic --max-delay-spread-s 3e-6 --output /tmp/decoded.npz
```

使用仓库提供的 MATLAB H5 接收向量进行分析和 CRC 解码（夹具没有噪声功率或真实信道元数据，因此使用 DMRS 信道估计，并对无噪声参考数据设置 `--noise-variance 0`）：

```bash
nr-pusch-rx --rx-config configs/rx_pusch_4ue.toml --input tests/fixtures/matlab_h5/RxTestVector.h5 --noise-variance 0 --channel-estimator dmrs --input-domain frequency --output /tmp/rx_test_vector_decode.npz
```

接收输出 NPZ 保存解码 bits 和逐用户 CRC 状态，旁边的 JSON 文件保存使用的配置及输入分析摘要。旧参数名 `--tx-config` 仍作为 `--rx-config` 的兼容别名。

### 网页外部接收分析

启动本地网页后，“接收分析”默认选用仓库内的 `configs/rx_pusch_4ue.toml` 和 `tests/fixtures/matlab_h5/RxTestVector.h5`，可直接启动默认 LMMSE 解码。也可上传 `.h5`/`.hdf5` 或 `.npz` 文件，选择其他 RX TOML 配置，或粘贴完整 TOML 覆盖，再设置输入域、LMMSE/LMMSE-SIC/K-best/EP/MMSE-PIC 检测器及其参数。页面显示输入网格摘要、按 UE 着色的软 QAM 星座点、逐 UE CRC 状态，并提供 bits/CRC NPZ 与 JSON 清单下载。H5 当前按 MATLAB `FreqData/IQdataPdu` 频域格式读取；NPZ 时域数组为 `iq`，频域数组为 `grid`。单文件上传上限为 12 MiB，数据由本地 Web 服务处理。

```bash
nr-pusch-web --host 127.0.0.1 --port 8765 --config-dir configs --runs-dir runs/web
```

对提供的 `RxTestVector.h5`，噪声方差应设为 `0`（该夹具不含噪声功率元数据）；DMRS 信道估计下 LMMSE 的四个 UE 均通过 CRC。

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
- `tests/fixtures/`：后续放置 MATLAB H5 参考夹具
- `docs/`：支持范围、接口和夹具说明
