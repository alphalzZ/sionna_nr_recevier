# NR PUSCH Lab

一个小型、配置驱动的 5G NR PUSCH 发送与 IQ 分析仓库。首批目标为 4 用户上行 MU-MIMO，每用户 1 层；发送端通过独立 IQ 文件和运行清单与接收端解耦。

## 当前发送端

当前实现以 Sionna 2.0.1 的 `PUSCHTransmitter` 生成 PUSCH 数据资源网格，并支持 CP-OFDM 和 DFT-s-OFDM 波形输出。DFT-s-OFDM 对数据符号应用酉 DFT，并按 type-1 低 PAPR 序列生成 DMRS 后进行 OFDM 调制。由于 Sionna 2.0.1 的组合 PUSCH 发射器不支持 transform precoding，配置中的 DFT 预编码 MCS 会映射到同调制阶数和码率的原生 Sionna MCS，以复用其 TB 编码器。当前配置为 DFT-s-OFDM MCS table 1/index 20，目标码率 0.6015625，TB 大小 28,168 bit。

## CDL 信道

`NrPuschCdlChannel` 使用 Sionna TR 38.901 CDL A–E 模型处理 4 个独立的上行 UE 链路，每个 UE 使用 1 根发射天线；4 条链路在 4 天线 BS 接收端叠加。接收阵列形状、CDL 模型、载波频率、时延扩展和速度范围等参数位于 `configs/cdl_38_901_4x4.toml`。该 profile 使用单极化 2×2 接收阵列。

先生成发送 IQ，再将其作为独立输入应用信道：

```bash
nr-pusch-tx --config configs/pusch_4ue.toml --output /tmp/pusch_tx.npz --seed 7
nr-pusch-channel --config configs/cdl_38_901_4x4.toml --input /tmp/pusch_tx.npz --sample-rate-hz 18000000 --output /tmp/pusch_rx.npz
```

发送输入必须为 NPZ 中的 `iq` 数组，轴顺序为 `[batch,user,tx_antenna,sample]`。信道结果 `iq` 轴顺序为 `[batch,rx_antenna,sample]`，`per_user_iq` 为 `[batch,user,rx_antenna,sample]`。输出包含线性卷积产生的信道尾部；未加入噪声和路径损耗，默认保留 CDL 的实际归一化功率设置。

## 多用户接收和 BLER

`NrPuschRx` 使用 Sionna PUSCH TB 解码器、4×4 MIMO 检测，以及 DFT-s-OFDM 每数据符号的逆 DFT。信道估计可选 `dmrs`（当前单 DMRS、双 OCC 端口对 profile 的静态 slot LS）和 `perfect`（仿真 CDL 抽头理想 CSI 上界）。DMRS 模式假设一个 slot 内信道不变，适用于当前零速静态 CDL profile；高 Doppler 或多 DMRS 配置需扩展时频插值并用抓包参考继续验证。

检测器可选 `lmmse`、`lmmse-sic`、`k-best`、`ep` 和 `mmse-pic`，参数通过 `detector` / `detector_parameter` 配置，接收 CLI 也提供 `--detector` 和 `--detector-parameter`。`k-best` 先用频域 LMMSE 等化整体 MU 信道，再将输出做 IDFT；每个时域采样点使用零时延等效空间矩阵进行四流 K-best 搜索，并把剩余频率变化作为残余 ISI 协方差。`lmmse-sic` 按估计信道功率从强到弱处理 UE：LMMSE 解调后进行 TB 译码，仅在该 UE CRC 通过时重编码并重构其 DFT-s-OFDM 资源网格，再从接收频域信号中消除该用户并处理下一个 UE。单次接收的 JSON sidecar 会记录 SIC 检测顺序和每个 UE 消除前的 CRC 状态。BLER 配置中的 `detectors` 会按相同 seed、CDL、SNR 和 payload 顺序比较检测器。

Sionna 原生逐 RE 的 `k-best`、`ep` 和 `mmse-pic` 假设每个 RE 独立承载 QAM；本仓库的 `k-best` 已改为上述时域采样检测方法。EP 和 MMSE-PIC 仍是 DFT-s-OFDM 下的实验性逐 RE 基线。当前 K-best 零时延信道近似需通过不同 CDL、时延扩展和 SNR 曲线继续评估。

配置 SNR 扫描点、批大小、帧上限和停止错误数后运行：

```bash
nr-pusch-bler --tx-config configs/pusch_4ue.toml --channel-config configs/cdl_38_901_4x4.toml --simulation-config configs/bler_smoke.toml --output /tmp/pusch_bler.csv --device cpu
```

将 `device` 写为 `cpu`、`cuda`、`cuda:0` 或 `auto` 可从 TOML 控制运算设备；`configs/bler_4ue_cdl_gpu.toml` 提供显式 CUDA 示例。当前运行环境没有可用 CUDA 设备，显式选择 CUDA 会给出错误提示。

扫描会逐 SNR 点发射随机 transport blocks、通过 CDL、按每个接收天线的测得信号功率注入复 AWGN，再用 CRC 与 payload 比对统计 BLER。CSV 包含 SNR、BLER、CRC fail rate、BER 和样本数，JSON sidecar 保存配置及完整统计；BLER 将 CRC fail 或任何 payload bit 错误都计为 block error。`bler_smoke.toml` 是短时连通性配置，正式仿真应增加 `max_frames_per_snr` 和 `target_block_errors`。

接收端也可独立加载 NPZ：`iq` 使用 `[batch,rx_antenna,sample]` 轴；`perfect` 模式另需 `channel_taps`（`[batch,user,rx_antenna,time,tap]`），`dmrs` 模式从 PUSCH DMRS 估计 CSI。

```bash
nr-pusch-rx --tx-config configs/pusch_4ue.toml --input /path/to/capture.npz --noise-variance 0.001 --channel-estimator dmrs --detector lmmse-sic --max-delay-spread-s 3e-6 --output /tmp/decoded.npz
```

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
