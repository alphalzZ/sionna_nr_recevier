# 四用户 DFT-s-OFDM PUSCH 接收机：算法设计与优化记录

本文记录仓库**现有实现**、对四组 MATLAB 接收抓包的**实测结果**、已经试过但未保留的方案，以及后续实验的判据。重点是高码率 `RxTestVectorCase123427.h5`；文中的 4/4 均指**四个 UE 的本机 TB CRC 全部通过**，不是与 H5 中参考载荷逐 bit 相同。实现入口为 `src/nr_pusch/receiver.py`，运行参数分别在 `configs/rx_pusch_4ue.toml` 和 `configs/rx_pusch_4ue_mcs27.toml`。

## 1. 结论与适用边界

- 当前已保留的改进是**调整高码率抓包的 DMRS 有限抽头拟合窗口**，不是一个已完成的自适应波束检测器：窗口从默认下界 `l_min=-6` 移至 `-44`，`max_delay_spread_s=2e-6`，保持 MMSE-PIC、8 次迭代、阻尼 0.5、接收噪声方差 0.0003。该抓包由 **2/4 提升到 3/4 TB CRC**；ue2 的码块从 0/6 提升到 6/6，ue3 仍为 0/6。**4/4 尚未实现**。
- 2026-10-01 完成当前 CDL-A/4 UE/MCS 20 的预注册 tap-power LMMSE 配对 holdout：25/30 dB BLER 分别降低 12.52%/11.89%，两点配对 97.5% 单侧上界均小于 0，data-RE CSI NMSE 也下降；60 dB 未回归。**仅保留为显式 `dmrs-lmmse` opt-in**，现有 `dmrs` 默认与 H5 抓包 profile 不变；测量细节见 §3.5。
- 2026-09-29 一轮结构化诊断（约 150 组配置）未能把 ue3 推过 2/6 CB，并已排除 SNR、CSI 偏差、DMRS 端口映射、扰码、检测器选择、PIC/LDPC 轮数、`err_var` 标定、CP 截断与逐符号跟踪等方向；ue3 的 `|h|²`、`err_var/|h|²` 和导频拟合残差都是四 UE 中最优的，单流 QAM 距离也与能解码的 ue2 同档，却 0/6 CB。结论与测量见 §3.1，**下一轮应先核对数据本身而非继续调参**。
- 低码率抓包统一沿用 `rx_pusch_4ue.toml`，没有把高码率抓包专用的窗口强加给其他抓包：`RxTestVector.h5` 为 4/4，`Case11121314` 为 4/4，`Case78914` 为 3/4（ue0 未通过）。
- 当前支持 4 UE、每 UE 1 层、4 根接收天线和两组共享 comb/OCC。DMRS 估计器逐 occasion 拟合，在相邻 occasion 间按 OFDM 符号位置线性插值，首尾保持最近估计；`err_var` 用平方权重传播独立 occasion 的测量误差。
- H5 没有提供可靠的接收噪声测量值。profile 中的 `noise_variance` 是调试接收机时采用的模型参数，不是抓包测得的热噪声功率。H5 的 `ue*_tx_bits` 可能是自身 CRC 已失败的参考链路输出，因此对抓包以本机 CRC 判定成功；参考 bit error 仅供诊断。仿真中发送原始 bit 已知，才可直接核对 BER/BLER。

## 2. 输入、信号模型与处理链

### 2.1 数据入口和几何尺寸

MATLAB 的 `FreqData/IQdataPdu_real`/`_imag` 经 `src/nr_pusch/iq/matlab_h5.py` 合并；原始轴为 `[ofdm_symbol, rx_antenna, active_subcarrier]`，读取后为 `[rx_antenna, ofdm_symbol, active_subcarrier]`。CLI 添加 batch 和接收端口轴，送入 `NrPuschRx.receive_frequency_grid()` 的复数网格为 `[batch, 1, 4, 14, 600]`。时域 NPZ 则走 `NrPuschRx.receive()`，输入 `[batch, 4, sample]`，由 Sionna 接收机进行 OFDM 解调；两条路径后续共用 DMRS 估计和 MIMO 检测器。`perfect` 信道模式只用于具备真实 CDL taps/频响的仿真上界，不应用作 H5 抓包的实测成绩。

当前抓包的 14 个 OFDM 符号中有 1 个 DMRS 符号，13 个数据符号各有 600 个有效子载波。MCS 27 对应 64-QAM：每 UE 的 7,800 个 QAM 数据符号产生 46,800 个编码 bit；TB 长度 43,032 bit，分为 6 个码块，目标码率 0.92578125。低码率 MCS 20 的 TB 长度为 28,168 bit、4 个码块。采样率由配置网格确定，此处为 18 MHz。DFT-s-OFDM 发射器对每个数据 OFDM 符号使用酉 DFT；接收端必须在频域均衡后按**相同的数据 RE 顺序**作酉 IDFT，再生成 QAM bit LLR。

可用简化的单子载波模型描述接收数据：

$$
\mathbf y_{s,k}=\mathbf H_{s,k}\mathbf x_{s,k}+\mathbf n_{s,k},\qquad
\mathbf y_{s,k}\in\mathbb C^4,\quad \mathbf H_{s,k}\in\mathbb C^{4\times4}.
$$

这里 $s$ 是 OFDM 符号，$k$ 是有效子载波，$\mathbf x$ 是 4 个 UE **经 DFT 扩频后的**频域数据。该逐子载波模型近似忽略不充分 CP 带来的 ISI、符号内时变造成的 ICI、各 UE 的独立时频偏和射频非理想；因此换检测器并不必然能补救 CSI/模型失配。

处理顺序：H5/NPZ 输入 → DMRS 有限抽头联合 LS 估计 $\widehat H$ 和估计误差方差 → 逐子载波 LMMSE 初始均衡 → 按数据映射顺序酉 IDFT → QAM 软解映射 → MMSE-PIC 迭代 → 按 UE 解扰、LDPC/码块 CRC/TB CRC。`--cb-crc` 额外暴露原本会被 TB 解码器丢弃的各码块 CRC 判定；输出的 `crc_status` 为 `[batch,user]`。

### 2.2 DMRS：共享 comb/OCC 的双用户有限抽头拟合

`DftSOfdmDmrsEstimator` 使用仓库实际 DFT-s-OFDM 发射器生成的低 PAPR DMRS 模板，而不是 Sionna 原生 CP-OFDM PUSCH 导频模板。它按 pilot RE support 自动组成两组 UE；每组两个 UE 共用 comb，伙伴端口在相邻两 pilot RE 上相对基准端口取 `+1,-1` 的 OCC 覆盖。初始化会验证组数、support 长度与这一相位关系；不能把四 UE 的导频当作互不干扰的四组普通 LS 导频。

设有效子载波数 $K=600$，$k_c=k-K/2$，候选延迟抽头 $\ell\in[l_{\min},l_{\max}]$；当前基函数是

$$
B_{k,\ell}=\exp\!\left(-j2\pi k_c\ell/K\right),\qquad
H_{r,u}[k]\approx\sum_{\ell}h_{r,u}[\ell]B_{k,\ell}.
$$

对同一 comb 的两个 UE $u,v$，已知 pilot 模板 $P_u[k],P_v[k]$，在该 comb 的 pilot 集合 $\mathcal P$ 上按列构造

$$
A_{k,(u,\ell)}=P_u[k]B_{k,\ell},\quad
A_{k,(v,\ell)}=P_v[k]B_{k,\ell},\quad
\widehat{\mathbf h}_r=\arg\min_{\mathbf h}\|\mathbf y_{r,\mathcal P}-A\mathbf h\|_2^2.
$$

代码以 `torch.linalg.lstsq` **联合**求解每个 DMRS occasion 上两个 UE 在每根 RX 天线上的抽头，用 $B$ 合成全部有效子载波的频响，再按相邻 DMRS occasion 线性插值到数据符号。用 $\operatorname{pinv}(A^HA)$ 的用户块、基函数与输入噪声方差计算每个 occasion 的逐频点 CSI 误差方差，并按插值权重平方传播。每组导频 RE 数必须至少覆盖两个 UE 合计的抽头数：代码约束为 $|\mathcal P|\ge2N_{\rm tap}$；窗口越宽，拟合自由度与噪声放大风险也越高。该误差方差不包含时间插值模型误差，也不保证覆盖 ICI、模型外多径或未建模的符号内变化。

`l_max` 使用 Sionna 的 `time_lag_discrete_time_channel(sample_rate_hz,max_delay_spread_s)` 确定；在本抓包 18 MHz 下，2 µs 对应 `l_max=42`，所以高码率 profile 的拟合窗口是 **[-44,42]，共 87 个抽头/UE**。默认 `l_min=-6`、3 µs 则是 [-6,60]。改变 `l_min` 不等于事后给现有 CSI 乘一条相位斜坡：它直接改变了**联合拟合的列空间**，会改变两个共享 comb 用户的分离、插值偏差和噪声传播。这是此次 ue2 恢复的可重复配置变化；无法仅凭 CRC 将原因唯一归结为某一条物理路径。

可选的 `estimate_delay` 是输出诊断：从频响 FFT 峰值和抛物线插值计算每个 UE、每根接收天线的 bulk offset，**不修正 CSI**，因此单独开启不应改变译码输出。sidecar 同时保留兼容字段 `estimated_bulk_delay_samples`（最后一个 UE）和完整的 `estimated_bulk_delay_samples_by_user`。

### 2.3 频域 LMMSE、逆扩频和 MMSE-PIC

`DftSOfdmMimoDetector` 首先调用 Sionna 的逐子载波 `LMMSEEqualizer`，输入 $Y,\widehat H$、CSI 误差方差和噪声方差，得到四 UE 扩频符号估计及有效噪声方差。接收机从 Sionna 的数据 RE 索引取同样顺序的 13 个完整数据 OFDM 符号，对每符号 600 个频域估计作 `ifft(..., norm="ortho")`，按符号平均子载波有效噪声方差，最后采用 APP QAM demapper 输出 bit LLR。这一方差平均是近似：频点噪声不等方差时 IDFT 后会出现相关噪声。

`DftSOfdmMmsePicDetector` 以此为第 0 轮 LLR。每一轮将 bit LLR 转为 QAM 点概率，再算软均值 $m_{u,s,n}=E[x_{u,s,n}]$ 与方差 $v_{u,s,n}=E[|x-m|^2]$。对每数据符号酉 DFT 得到 $\widetilde m_{u,s,k}$，由当前 CSI 计算其在各 RX 天线上的软干扰。对目标 UE $u$，保留它自己的分量，抵消其他 UE 的软均值：

$$
\mathbf y^{(u)}_{s,k}=\mathbf y_{s,k}
-\sum_j\widehat{\mathbf h}_{j,s,k}\widetilde m_{j,s,k}
+\widehat{\mathbf h}_{u,s,k}\widetilde m_{u,s,k},\qquad
\widehat x^{(u)}_{s,k}=
\frac{\widehat{\mathbf h}_{u,s,k}^{H}\mathbf y^{(u)}_{s,k}}
     {\|\widehat{\mathbf h}_{u,s,k}\|^2}.
$$

迭代阶段在该软抵消结果上采用归一化匹配滤波，而**不是**再次求解完整的 4×4 LMMSE 系统。有效方差用热噪声、各流软符号不确定性及 CSI 误差的近似二阶矩叠加；按 $|\widehat h_{r,u}|^2$ 加权后除以信道模平方，再在数据符号的频点上平均。酉 IDFT 后重新软解映射。令 $\alpha=$ `detector_damping`，更新为

$$
L^{(t+1)}=(1-\alpha)L^{(t)}+\alpha\,L^{(t+1)}_{\rm new}.
$$

高码率 profile 使用 8 轮、$\alpha=0.5$。这里的反馈来自**当前解调器的 QAM 软信息**，并没有把 LDPC 的外信息迭代反馈到检测器；“增加 PIC 迭代次数”并不等于进行了 Turbo 均衡。

#### `soft-mmse-pic`：LDPC 外信息反馈检测器

新增的 `DftSOfdmSoftMmsePicDetector` 是独立可选项，不改变 `mmse-pic` 的算式或既有 profile。其第 0 轮仍是上述频域 LMMSE、酉 IDFT 和 APP QAM 解映射；每个外循环把当前检测 LLR 交给单独的软输出 `LDPC5GDecoder`，不调用硬输出 TB 解码器取 bits。软反馈复用最终 TBDecoder 的编码器、解扰序列和 `_output_perm_inv`：先解扰，把 punctured codeword 补零并逆交织到码块位序，再按码块运行 BP。回馈量为

$$
L_{\rm LDPC,ext}=L_{\rm LDPC,post}-\operatorname{clip}(L_{\rm detector},-20,20),
$$

即减去本轮已使用的通道证据，而不是把译码后验原样重复累加；反馈恢复到交织、加扰的 PUSCH coded-bit 位序后限幅。标准 RNTI 和显式 H5 扰码都共享该流程。4-user MCS 20/27 profiles 分别走 4/6 个 CB；当前两种 profile 的 Sionna rate-matched 长度都没有额外填充位，其他编码配置仍按 TBDecoder 的零填充约定处理。

外信息先以 `detector_damping` 与上轮先验作线性阻尼；其他 UE 的符号概率由其本轮检测器 LLR 加阻尼外信息构成，避免把单独的 LDPC 外信息误当完整符号后验。对每个目标 UE，先从接收向量中减去其他 UE 软均值，再使用包含热噪声、其他 UE 残余方差及 CSI 误差的逐天线对角协方差 $C_{-u}$ 加权：

$$
g_u=\sum_r\frac{|h_{r,u}|^2}{C_{-u,r}},\quad
z_u=\frac{\sum_r h^*_{r,u}y^{(u)}_r/C_{-u,r}}{g_u},\quad
\sigma_u^2=g_u^{-1}.
$$

该单流 LMMSE 不把目标 UE 先验再次缩入估计；每个扩频组平均方差后作酉 IDFT 和 APP 解映射，所得 detector LLR 送入下一外循环。默认 `detector_parameter=1` 表示一次 LDPC→检测反馈，`detector_damping=0.25`；每次内部 BP 与最终 TBDecoder 的迭代数仍由 `num_decoder_iterations` 决定。接收器最后只由原 TBDecoder 对最终 detector LLR 做一次硬判及 TB/CB CRC，sidecar 分别记录反馈轮数、阻尼和算法说明。低码率 MATLAB 抓包默认参数为 4/4 CRC；高码率结果按实际 CB/TB CRC 单独记录如下，不以参考 bit errors 代替 CRC，也不据单帧结果宣称普遍 BLER 改善。

Web RX 会将已选 profile 或粘贴的 `[receiver]` TOML 同步到检测器控件。`soft-mmse-pic` 未配置 `detector_parameter` 或 `detector_damping` 时，界面显示实际默认值 `1` 和 `0.25`；显式配置值保持不变并随解码请求提交。

对 `RxTestVectorCase123427.h5` 使用既有高码率 profile 的 `l_min=-44`、`max_delay_spread_s=2e-6`、噪声方差 `0.0003` 和显式扰码，单帧实测：默认 1 轮/0.25 为 **2/4 TB CRC**（ue0/ue1 通过；ue2 3/6 CB、ue3 0/6），4 轮/0.25 为 **3/4 TB CRC**（ue0–ue2 各 6/6 CB；ue3 0/6）。旧 `mmse-pic` 8 轮/0.5 在该 profile 是 3/4。4 轮只恢复了已知可通过的前三个 TB，不改善 ue3；这是单抓包结果，不能推出配对 BLER 增益。默认仍按算法计划保留 1 轮，比较其他反馈轮数应显式传入 `detector_parameter`。

### 2.4 TB 判定、显式加扰与可选 SIC

Sionna `TBDecoder` 对 MIMO detector 的 46,800 bit/UE LLR 解扰、逆交织/码块拆分、LDPC 译码、去码块 CRC，最后验证 TB CRC。H5 如另提供 `ue<k>_scrambSeq`，CLI `--scrambling <h5>` 按 `[4,46800]` 读取并验证二值性，用该序列替换按 RNTI 推导的解扰序列；**不能**用不相符的 RNTI 扰码尝试判断信道算法好坏。LMMSE-SIC 的内部 CRC 门控解码器和重编码器也使用同一序列，未提供时才使用标准 RNTI 路径。CRC 真值与捕获 H5 中所谓“参考 tx bits 是否相同”是两个不同量。

`DftSOfdmLmmseSicDetector` 是另一个已有检测器：按估计的信道功率强到弱处理 UE，仅当内部 TB CRC 通过才重编码该 UE、按估计 $H$ 重构接收资源网格并相减，同时从后续系统中去掉该 UE 信道列。错误重编码或虚假的 CRC 状态会污染后续 UE，因此这里的内部扰码必须正确。K-best 与 EP 则用 LMMSE 前端、IDFT 之后的零时延等效空间信道与彩色残余 ISI 近似进行检测；它们不显式联合处理所有跨采样 ISI，不能把更大的搜索宽度/迭代数等同于 CSI 改善。

## 3. 高码率抓包：优化过程与负面结果

以下为单帧抓包的实验记录，不是跨帧统计显著性结论。除非另列，输入均为 `RxTestVectorCase123427.h5`、对应 `scrambSeqCase123427.h5`、MCS 27、频域网格、CPU、`noise_variance=0.0003`，判据为本机 TB/CB CRC。尝试方案中有一次性 `/tmp` 探针；**只有 profile 调整和 SIC 显式加扰修复已并入产品代码**，不要把表中实验算法当作目前可用的 `detector` 选项。

| 阶段 / 方向 | 方案与观察 | 结论及保留状态 |
| --- | --- | --- |
| 原始高码率基线 | `l_min=-6`、约 3 µs 拟合窗、MMSE-PIC 8 / 0.5：ue0/ue1 TB CRC 通过，ue2 0/6 CB、ue3 1/6 CB，合计 **2/4 TB**。原先在 3/6 µs、不同噪声和五种检测器上扫描，最高 2/4。 | 基线；先核对扰码、MCS/TB 长度和 DMRS 端口，避免把解扰失败误当信道失败。 |
| 延迟支持扫描（保留） | 高码率 profile 改为 `l_min=-44`、`max_delay_spread_s=2e-6`，其余保持不变。ue0/ue1/ue2 各 6/6 CB、TB CRC 通过；ue3 0/6，合计 **3/4 TB**。 | 修改联合 DMRS 拟合支持区间后 UE2 恢复；已保留，不能称为 4/4。该区间是该抓包的调优值，不应静默套用于其他抓包。 |
| 每 UE 不同窗口（未保留） | 探针让 ue0–2 使用 `[-44,42]` 估计，让 ue3 使用 `l_min=-14`、4 µs 的另一估计，然后组合 CSI 输入同一 MMSE-PIC。结果 **[6,6,6,5] CB、[通过,通过,通过,失败] TB**；ue3 仅 CB0 失败。 | 说明 UE3 对 CSI 模型敏感，但按 UE 拼接固定窗口仍非 4/4，且易过拟合此单帧。值得后续研究**由 DMRS 数据选择**每 UE 支持窗，而非硬编码 3 号 UE。 |
| 检测器和噪声参数 | 扫描 LMMSE、LMMSE-SIC、K-best、EP、MMSE-PIC，以及 PIC 轮数/阻尼、噪声假设。原窗口最高 2/4；调整后的配置下 LMMSE-SIC 加正确显式扰码为 2/4，MMSE-PIC 为 3/4。 | 单换检测器不能证明消除了 CSI 偏差；EP/K-best 的零时延空间近似与软信息标定各有额外限制。 |
| 波束分组与 CRC 门控抵消（未保留） | 测试每 UE 主接收天线分配、0/1 与 2/3 波束对、按已通过 CRC 的 TB 重编码并拟合有限抽头泄漏后抵消。试验性 beam-SIC 在调优 profile 下仅 **2/4**，比 MMSE-PIC 的 3/4 差；有版本把已抵消用户的 CSI 列直接置零，产生非有限 LLR。 | 已删除实验检测器；不能靠置零列“屏蔽”流而继续使用要求非奇异有效噪声的等化/解映射。以后应实现真正的活动流子系统并保留 CRC 验证。 |
| 已验证用户的波形重构 / 数据辅助 LS（未保留） | 通过 CRC 的 UE 可以用同一扰码重编码；尝试以部分后续数据 OFDM 符号的 RE 做多用户有限抽头重拟合，再与初始 CSI 混合，或重做检测。观察到训练残差可下降，**没有得到 ue3 的 TB CRC**，较强混合还使其他 UE 失效。 | 拟合残差下降不是译码改善的充分条件；低置信 UE 的判决不可作为无误导频，且必须做未参与拟合的 RE/符号验证。 |
| 早期符号跟踪 / 单用户抵消（未保留） | ue3 的混合窗口版本只剩首个 CB 失败；尝试前几个 OFDM 符号的判决引导幅相调整、符号 LLR 重加权，以及重构并抵消 ue0–2 后对 ue3 用不同 RX 天线组合做单流匹配滤波/解映射。 | 实验均未使 ue3 TB CRC 通过；不能据此断言物理上没有早期符号失真，仅说明这些模型和参数没有修好。 |
| 延迟补偿、空间投影、CPE（未保留） | `--estimate-delay` 只输出峰值诊断不改 CSI；直接把估计 offset 作为相位斜坡补偿曾使低码率正常抓包从 4/4 降到 1/4。历史秩一空间投影使两个原 4/4 抓包降为 3/4；对应实现现已删除。盲四阶矩/判决引导 CPE 原型也曾使 4/4 降为 3/4，并出现非有限值。 | 不要仅凭某一接收天线主径就假定每 UE 4 天线信道严格秩一；相位补偿需与拟合窗/分数时延模型一致。CPE 原型未合入。 |
| LDPC / CRC 辅助探索（未保留） | 在混合 CSI 的 ue3 CB0 上尝试不同 LDPC 校验节点更新（boxplus、minsum 等）、flooding/layered、10–160 轮，CB0 仍未通过；对低置信 bit 的小规模 CRC 列表翻转，无候选通过最终 TB CRC。 | 先提高 CB0 检测软信息可信度；提高 LDPC 轮数、仅调整 LLR 尺度或凭 CB CRC 匹配并不能保证 TB 正确。 |
| 每 comb 对独立窗口（未保留） | 一次实现 per-pair `l_min`/`l_max` 扫描：固定 pair(0,1) 为 `[-44,42]`，对 pair(2,3) 扫 58 组（`l_min∈[-90,-6]`、`l_max∈[12,52]`）。最好仍是 **3/4**，ue3 最高仅 2/6 CB；`noise_variance` 扫 1e-5…6e-2 时 3e-4 为唯一最优点。 | 窗口这一自由度已用尽：无法靠选窗把 ue3 推过 2/6 CB。实验性 per-pair 旋钮已删除。 |
| 每 UE 窗口会破坏 OCC 分离（已证伪） | 实现并验证了 per-UE `l_min`/`l_max` 管线（顺带修掉 `self._num_ofdm_symbols` 漏写的回归）。控制实验（四 UE 全同窗）精确复现基线 3/4 与 2/4，说明管线正确；但**同一 comb 对内两 UE 用不同窗口**（如 ue2 `[-10,22]`、ue3 `[-44,42]`）会让四个 UE 一起掉到 **0/4**，而 ±1…±8 抽头的小扰动行为正常。 | OCC 的 `±1` 覆盖在延迟域等价于 ±150 bin 平移，只有两 UE 共享同一抽头窗时两块设计列才正交。**这否定了 §5 第 2 条「每 UE 独立支持窗」的提法**：选择粒度最多到 comb 对。该管线已删除。 |
| DMRS 端口全排列（未保留） | 24 种 `dmrs_port` 分配全试：恒等 `(0,1,2,3)` 最好 3/4；**ue3 在全部 24 种排列下都是 0/6 CB**。 | 排除 DMRS 端口/用户映射错误。 |
| 扰码行置换（无效判据） | 用 `scrambSeq` 的其他行替换 ue3 的扰码行，四种取值结果完全相同（3/4，ue3 0/6）。 | 解扰是码字保持变换，**扰码选择按构造不可能影响 CRC**，不能作为信道算法好坏的判据；此前若以扰码试验论证问题位置，结论无效。 |
| Oracle CSI 替换（诊断） | 用 ue0–2 的**逐符号块 LS oracle 信道**替换其 DMRS 估计（B=12/20/30/50，ue3 仍用 DMRS 估计）重跑译码：仍为 3/4、ue3 0/6。 | 失败**与 ue0–2 的 CSI 误差无关**，问题在 ue3 自己的信道列；推翻「其他用户 CSI 误差泄漏进 ue3」的假设。 |
| CP 截断（未保留） | 周期信道模型按 ±4…±44 抽头截断后重跑：只有无截断的 ±44 达到 3/4，其余 ≤2/4，ue3 始终 0/6。 | 截断模型外的 tap 不能救回 ue3，ICI 截断不是本抓包的原因。 |
| `err_var` 放大（未保留） | CSI 误差方差统一放大 ×0…×128，以及只放大 ue3 ×2…×64：**全部 ≤ 基线**，放大只会让 ue2 退化。 | 当前 `err_var` 未低估；等化器不是因为过度自信而失败。 |

上表比较的是固定 H5 抓包的**同一组接收样本**；改变 `noise_variance` 改的是接收机的噪声假设，不是在 H5 上重新加噪。另一方面，跨独立仿真运行的 BLER 结果不一定使用逐帧相同的随机信道和噪声；如要做配对比较，须控制完整扫描配置、随机种子和 batch 分组。`README.md` 中的历史 BLER 曲线不应直接解释为某一修改在这个 H5 上的因果证据。

### 3.1 本轮结构化诊断（2026-09-29）

用一次性 `/tmp` 探针在真实代码路径上做了结构化诊断，得到若干**推翻既有假设**的结论。判据仍为本机 CRC，但下列测量本身与 CRC 无关。

**1. 本抓包的四 UE 近似空间正交，不存在参考文档描述的旁瓣耦合。** DMRS 估计的逐天线功率为 ue0 `[0.0490, 0.0003, 0.0002, 0.0002]`、ue1 `[0.0004, 0.0669, 0.0004, 0.0002]`、ue2 `[0.0003, 0.0002, 0.0384, 0.0003]`、ue3 `[0.0001, 0.0001, 0.0004, 0.0654]`。交叉响应比主瓣低约 **24 dB**，4×4 信道条件数中位数 1.00、最小奇异值比 0.35–0.65。联合检测在本抓包几乎没有增益，`paste-2.md` 的 `b_ij` 旁瓣泄漏模型不描述这组数据。

**2. ue3 不是 SNR 受限，其单流质量与能解码的 UE 同档。** 对每个 UE 用**同一套流程**（用其 DMRS 信道抵消其他已解码用户 → 用自己的 DMRS 信道匹配滤波 → 酉 IDFT → 到最近 64QAM 点的距离）测得：ue0 `0.1030`（仍有 ue3 干扰）、ue1 `0.0734`、ue2 `0.1276`、**ue3 `0.1050`（零干扰）**。ue3 在零干扰下的距离优于 ue2，而 ue2 是 6/6 CB 通过。**不能再用「ue3 太弱 / 旁瓣干扰太强」解释 0/6。**

**3. ue3 的 CSI 各项指标都是四者中最好的。** `|h|²` 为 `1.65e-2`（最大），`err_var/|h|²` 为 `0.0026`（最小），pair(2,3) 的导频联合拟合残差为 `0.0024`（最干净）。它的 LLR 幅度（均值 46.8）甚至高于 ue0（35.8），且在 13 个数据符号上均匀分布。**信号很强、CSI 很准、LLR 很强，却 0/6 CB**——这与「CSI 偏差」类解释不符。

**4. 由已解码用户构造的「真值」在 antenna 3 上不可用。** 早期探针用 ue0–2 的逐符号块 LS 当 oracle，但在 antenna 3 上 ue3（唯一未解码用户）占主导，其干扰被块 LS 吸收进 ue0–2 的「信道」，导致 antenna 3 上出现 0.99 的虚假相对误差，而 ue0–2 各自主瓣天线上只有 0.09。**该 0.99 是探针伪迹，不是 DMRS 估计误差**；不要再用它在共享天线上做 CSI 判据。相应地，所谓「逐符号 35–50% 的信道起伏」在三个 UE 之间的互相关只有 0.00–(-0.04)，且对能解码的 ue0–2 同样存在，说明它是块 LS 吸收残余干扰/ISI 的产物而非真实时变；静态逐符号 CSI 模型与三个 UE 通过 CRC 并不矛盾。

**5. 判决引导恢复收敛到同一个平庸不动点。** 取消 ue0–2 后对 ue3 做判决引导 LS，从匹配滤波初值（QAM 距离 53.2）和从 DMRS 初值（0.128）**两个相差极大的起点**出发，都单调收敛到 ≈0.14 后停滞。说明数据本身不支持明显更好的信道估计。

**6. 导频残差会骗人，必须用留出判据。** pair(0,1) 在 antenna 3 上有 **17% 的导频功率无法被 `[-44,42]` 模型解释**；放宽到 `[-104,42]`（147 抽头，仍在 `2N_tap ≤ |P| = 300` 预算内）残差降到 0.0092，但**译码反而降到 2/4**——147×2 = 294 自由度对 300 个观测已接近插值拟合。改用**按完整相邻 OCC RE 对划分的 4 折留出预测误差**后，最优窗口在 (pair, 天线) 之间差异极大（pair(0,1) 的 ant0/ant1 选 `[-4,22]`，pair(2,3) 的 ant2 选 `[-10,22]`、ant3 选 `[-50,52]`），与 in-sample 残差最小的窗口**完全不同**。**进一步的窗口工作必须用留出判据，且选择粒度不能细于 comb 对。**

### 3.2 抓包调优参数能否迁移到 CDL 仿真（2026-09-29）

**问题**：把抓包链路的 MMSE-PIC 配置搬进 BLER 仿真链路，BLER 会变好还是变差？

**方法**：在 `configs/pusch_4ue.toml`（MCS 20，与低码率抓包 profile 同码率）+ `configs/cdl_38_901_4x4.toml` 上做**配对 A/B**。按 §3 的复现性要求，四个变体固定 `seed = 20260924`、SNR 网格 `[25,30,35,40]`、batch 20、每点 `max_frames_per_snr = 1000`（4,000 个 TB/点），并关闭 `target_block_errors` 早停与 `stop_at_zero_bler`，使各变体消耗**完全相同**的载荷/信道/噪声随机流；只改检测器旋钮。信道配置未改，因此 CDL 抽出的信道逐帧相同。A 与 A' 是同一配置的重复跑。

| 变体 | MMSE-PIC 迭代 / 阻尼 | 接收窗 `max_delay_spread_s` | 25 dB | 30 dB | 35 dB | 40 dB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **A 原仿真基线** | 4 / 0.25 | 跟随信道（3 µs） | 0.5725 | 0.0877 | 0.0035 | 0.0003 |
| A' 重复跑（配对校验） | 4 / 0.25 | 跟随信道（3 µs） | 0.5725 | 0.0877 | 0.0035 | 0.0003 |
| **B 对齐 `rx_pusch_4ue.toml`** | 8 / 0.5 | 6 µs | 0.6690 | 0.2567 | 0.0963 | 0.0127 |
| C 仅迭代+阻尼 | 8 / 0.5 | 跟随信道（3 µs） | 0.5560 | 0.2500 | 0.0700 | 0.0043 |
| D 仅窗口 | 4 / 0.25 | 6 µs | 0.6787 | 0.1455 | 0.0050 | 0.0010 |

**配对成立**：A 与 A' 的 `transport_blocks` 与 `bit_errors` 在每个 SNR 点逐 bit 相同，随机流可复现，因此上表差异只能来自检测器配置。

**结论：不是提升，是退化，而且幅度很大。** B 相对 A 的 BLER 比为 30 dB **2.9×**、35 dB **27.5×**、40 dB **51×**。分解后主因是 **8 次迭代 + 阻尼 0.5**（C 单独即使 35 dB 差 20×、40 dB 差 17×），6 µs 接收窗次之（D 单独在 35 dB 只差 1.43×，40 dB 差 4×）。BER 同向：35 dB 由 2.68e-05 升到 2.31e-03。

**这说明抓包调优参数不能直接迁移。** §3 里 `l_min = -44` 之类取值是对**单帧抓包**的补偿，而 MMSE-PIC 的 8 次迭代 / 阻尼 0.5 在 30 kHz SCS 的 14 符号 DFT-s-OFDM 块上过度迭代、在长循环中累积软符号自干扰。两者都是「对该数据集有效」，不是「对该算法普适」。已据此**回退** `configs/bler_4ue_cdl.toml` 与 `configs/bler_4ue_cdl_gpu.toml` 的检测器数值，只保留新增的接收窗旋钮；`tests/unit/test_simulation_config.py` 同时固定两件事：出厂仿真配置保持自身基线，以及抓包 profile 的取值可以通过 `l_min` / `max_delay_spread_s` / `detector_parameters` / `detector_damping` 逐项复现。

### 3.3 Soft-MMSE-PIC 配对 CDL BLER（2026-09-29）

在 GPU（RTX 3060 Laptop，6 GiB）上按相同 seed `20260924`、SNR `[25,30,35,40]` dB、batch 8、每点 1,000 frames（4,000 TB）、20 次 BP、`target_block_errors=10^9`、`stop_at_zero_bler=false` 比较 `mmse-pic` 4/0.25 与 `soft-mmse-pic` 1/0.25。两者使用同一 MCS 20 TX/CDL、频域信道与 DMRS CSI；旧检测器全量重复跑后每点 `bit_errors` 完全一致。

| SNR (dB) | 旧 MMSE-PIC BLER / BER | Soft-MMSE-PIC BLER / BER | 旧 runtime (s) | Soft runtime (s) |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 0.585 / 4.9549e-2 | 0.386 / 3.2080e-2 | 86.70 | 122.31 |
| 30 | 0.07825 / 4.1867e-3 | 0.0185 / 9.5304e-4 | 86.21 | 121.54 |
| 35 | 0.0045 / 7.4757e-5 | 0.00025 / 7.7393e-6 | 87.03 | 123.20 |
| 40 | 0.00075 / 3.5501e-8 | 0 / 0 | 88.22 | 124.05 |

在该配对运行中，soft detector 的每点时间约多 35–43 秒；重复旧扫描的运行时间明显波动（约 218–328 秒/点），因此不据此推断稳定吞吐差异。另一个独立 batch-8、8-frame、25 dB GPU 探针测得峰值 allocated memory 为旧 933.4 MiB、新 909.5 MiB，peak reserved 为 1,784/1,886 MiB；该短探针不是 1,000-frame sweep 的峰值显存。此配对曲线支持在 CDL 上继续保留该独立检测器，但不足以修改默认 detector/profile；固定 H5 上 default 1 轮只得 2/4，4 轮得 3/4，ue3 仍失败，单帧捕获与 BLER 曲线结论分开。

`configs/bler_4ue_cdl.toml` 与 `configs/bler_4ue_cdl_gpu.toml` 的多检测器扫描现包含 `soft-mmse-pic`：CPU 使用 1 轮 LDPC 外反馈、batch 2；GPU 配置使用 4 轮、batch 8；两者阻尼均为 0.25。默认检测器仍为 `lmmse-sic`，旧 `mmse-pic` 保持 4 轮；全量扫描仍按各自配置的 SNR、帧数和停止条件运行，与上表的配对条件不完全相同。

### 3.4 空间签名/投影实验记录（实现已移除）

以下是历史测量；对应的 estimator API、CLI 选项、实验 runner、空间投影配置与专用测试已清理，当前仓库不再提供空间投影功能。原始独立 holdout 记录文件不在当前树中；可用的历史测量摘要与结论保留在本节下方。

- 20–30 dB、每点 8 帧的初筛中，基线 block errors 为 32/32、22/32、5/32；低秩投影和秩一弱投影均退化，秩 4（等价于不投影）接近基线。
- 25 dB 扩展到 128 TB 后，基线 88/128 TB 错误；全带秩一、10-RB 与 25-RB 分组均为 128/128 错误。分组没有挽回性能。
- 独立 learned-basis holdout 中，25 dB 基线与候选均为 6,077/16,000 TB 错误，30 dB 均为 367/16,000；候选接受率为 0%，因此实际输出与基线完全相同。
- 60 dB 安全检查中，带回退保护的各配置均为 0/64 错误；这只表明该样本没有回归，不是投影收益。未加保护的低秩投影则明显破坏高 SNR 解码。

结论：测得的 BLER/BER 没有改善；低秩投影在部分条件下显著退化，保护后的 learned-basis 方案则完全回退。保留实验结论，不保留无收益的空间投影代码和测试。

### 3.5 CDL-A tap-power LMMSE（预注册配对 holdout，2026-10-01）

本轮评估的是仿真信道估计，不是 §3 的 H5 抓包调参。TX/CDL 固定为 `configs/pusch_4ue.toml` + `configs/cdl_38_901_4x4.toml`（CDL-A、4 UE/4 RX、MCS 20）；单符号 DMRS、端口、OCC、接收窗 `l_min=-6` / `max_delay_spread_s=3e-6` 均未改变。baseline、candidate、perfect-CSI 上界共用每帧 payload、CDL realization 和 AWGN，只切换 CSI estimator；检测统一为 `soft-mmse-pic` 1 轮、阻尼 0.25、20 次 BP。

candidate 使用联合 OCC 设计矩阵 $A$ 与对角抽头先验 $R=\operatorname{diag}(p)\oplus\operatorname{diag}(p)$：

$$
\widehat{\mathbf h}=RA^H(ARA^H+\sigma^2I)^{-1}\mathbf y,\qquad
C_{\rm post}=R-RA^H(ARA^H+\sigma^2I)^{-1}AR.
$$

频域 `err_var` 由 $C_{\rm post}$ 投影到现有 DMRS 基底得到；零噪声退回满秩 LS。先验由独立 seed `21260924` 的 256 个 CDL realization 拟合，每 tap 设 `max(p)*1e-8` 正下限；训练不读取 development/holdout channel、payload 或 CRC。开发集 512 帧/SNR（seed `20260924`），holdout 3,000 帧/SNR（seed `22260924`，25/30 dB），另以 60 dB、64 帧检查高 SNR；BLER 差值按整帧重采样，bootstrap seed `20261001`、10,000 次。

| SNR | DMRS BLER | LMMSE BLER | 相对降低 | candidate−baseline BLER 的 97.5% 单侧上界 | data-RE CSI NMSE（DMRS → LMMSE） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 dB | 0.36933 (4432/12000) | 0.32308 (3877/12000) | 12.52% | −0.04200 | 0.00201394 → 0.00151686 |
| 30 dB | 0.02383 (286/12000) | 0.02100 (252/12000) | 11.89% | −0.001833 | 0.00063889 → 0.00056701 |

预注册门槛全部通过；60 dB 两臂均为 0/256 TB error。结果只支持该静态 CDL-A 配置上的显式 opt-in，不证明对其他 CDL 或捕获 IQ 泛化，也不替代 perfect CSI 上界。选择 `dmrs-lmmse` 时，BLER/RX/Web 按信道配置在共享先验目录中按兼容性键查找该 prior，并校验其元数据与当前 TX/CDL/抽头窗/FFT/采样率一致；缺失、不兼容或缺少验收发布标记时报错，不回退到 LS。

正式运行（无 smoke 覆盖）：

```bash
nr-pusch-estimator-validation --tx-config configs/pusch_4ue.toml --channel-config configs/cdl_38_901_4x4.toml --validation-config configs/channel_estimation_validation.toml --output /tmp/channel_estimation_validation.json --device cuda:0 --prior-dir configs/tap_power_prior
```

JSON 同时记录 aggregate/per-UE BLER、CRC failure、BER、data-RE NMSE、运行时间和 perfect-CSI gap。

RTX 3060 Laptop GPU（6 GiB）全量运行耗时约 74 分 56 秒。汇总、逐帧结果和诊断用训练 prior 分别保存于 `/tmp/channel_estimation_validation.json`、`/tmp/channel_estimation_validation.frames.npz`、`/tmp/channel_estimation_validation.prior.npz`；门槛通过时另有带验收标记的副本发布到 `--prior-dir`（默认 `configs/tap_power_prior/`）。连通性 smoke 使用独立输出和缩小帧数，不计入 holdout 结论。

## 4. 可复现验证与输出读取

以下在仓库根目录、仓库虚拟环境中执行，输出放 `/tmp`，不修改供应的 H5 文件：

```bash
PYTHONPATH=src /home/le-lei/workspace/test/.venv/bin/python -m nr_pusch.cli.rx \
  --rx-config configs/rx_pusch_4ue_mcs27.toml \
  --input tests/fixtures/matlab_h5/RxTestVectorCase123427.h5 \
  --scrambling tests/fixtures/matlab_h5/scrambSeqCase123427.h5 \
  --cb-crc --device cpu --output /tmp/case123427_decode.npz

PYTHONPATH=src /home/le-lei/workspace/test/.venv/bin/python -m unittest \
  tests.integration.test_matlab_rx_capture -v
PYTHONPATH=src /home/le-lei/workspace/test/.venv/bin/python -m unittest \
  discover -s tests -p 'test_*.py' -v
```

期望高码率 CLI 打印 `CRC pass: 3/4`，ue0/ue1/ue2 各 6/6 CB、ue3 0/6。`/tmp/case123427_decode.json` 中 `receiver.cb_crc_status` 保存每 UE 6 个布尔值，`reference_comparison.crc_status` 保存本机 TB CRC；`reference_comparison.bit_errors` 只比较抓包中**不可靠的参考链路载荷**。接收机还检查 soft LLR 有限性：非有限 LLR 应报错，不能将其解释为 CRC 通过。对 profile 外抓包调整 `--l-min`、`--max-delay-spread-s`、`--noise-variance` 前应记录这些覆盖值，否则结果不可复现。

回归基线：`tests/integration/test_matlab_rx_capture.py` 检查原始抓包全 4 UE 解码、高码率 profile 至少 ue0–ue2 CRC 通过、SIC 内部使用显式扰码进行 CRC 门控；`soft-mmse-pic` 有 LDPC 外信息位序/纠错单测、4×4 CDL payload/CRC 回环和 paired BLER 检查。历史上一轮加入 learned spatial projection 与配对 bootstrap 用例时，suite 共运行 64 项、63 项通过；唯一失败是 `test_matlab_payload_and_frequency_grid_match_transmitter`，因现有 `configs/pusch_4ue.toml` DMRS 端口映射与供应 MATLAB fixture 置换不一致。空间投影代码及其专用测试现已移除；上述数量仅为历史记录。未改动该用户配置或 fixture。

2026-10-01 本次完整 suite 运行 57 项，56 项通过；唯一失败仍为 `integration.test_pusch_transmitter.PuschTransmitterTest.test_matlab_payload_and_frequency_grid_match_transmitter`，供应 MATLAB fixture 与发射端频域网格有 1,200/33,600 项不匹配。与本估计器变更无关。
2026-10-02 本次完整 suite 运行 65 项、64 项通过；唯一失败仍是 `test_matlab_payload_and_frequency_grid_match_transmitter`，供应 MATLAB fixture 与发射端频域网格有 1,200/33,600 项不匹配。新增的多 DMRS 接收、映射、Web API 回归均通过；未改动供应 fixture 或活动 TX profile。

### CDL-A 多 DMRS occasion 仿真对比（2026-10-02）

使用 `configs/pusch_4ue.toml`、`configs/cdl_38_901_4x4.toml`（CDL-A、3.5 GHz、0 m/s、4×4）、MCS 20；其余固定为 LS DMRS、频域信道、`soft-mmse-pic` 一轮、20 次 LDPC 迭代、阻尼 0.25、CPU、seed `20260924`。只改变 `dmrs_additional_position`；0/1/2 分别使用 1/2/3 个 DMRS occasion。每点 `batch_size=2`，因此每帧 4 个 TB。

aggregate JSON 保存在 `/tmp/dmrs_interpolation_comparison.json` 与 `/tmp/dmrs_interpolation_comparison_25db_256frames.json`。

| Additional position | Bits/TB | 25 dB BLER / BER（256 帧，1024 TB） | 30 dB BLER（64 帧，256 TB） | 60 dB BLER（64 帧，256 TB） |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 28,168 | 0.40723 / 0.03438 | 0.03125 | 0 |
| 1 | 26,120 | 0.40430 / 0.03072 | 0.02734 | 0 |
| 2 | 23,568 | 0.32715 / 0.02599 | 0.01953 | 0 |

在 25 dB 样本中，position 2 相对 position 0 少 82/1024 个 TB CRC 失败（BLER 0.40723→0.32715）；position 1 为 414/1024 对 417/1024，BLER 基本持平但 BER 下降。30 dB 每组仅 256 TB，60 dB 全部为零错误；这些是同 seed 的有限样本观察，不是置信区间或统计 holdout，不能据此保证普遍 BLER 增益。额外 DMRS 会减少数据 RE：表中 bits/TB 分别下降约 7.3% 和 16.3%，故可靠性比较不等于等吞吐比较。高速移动场景未覆盖。

## 5. 下一轮优化设计与实验纪律

1. **先解释「强 LLR + 0/6 CB」这一矛盾，而非继续调参。**§3.1 已排除 SNR、CSI 偏差、端口映射、扰码、检测器、PIC/LDPC 轮数、`err_var`、CP 截断与逐符号跟踪（~150 组配置，ue3 最高 2/6 CB）。ue3 的 `|h|²`、`err_var/|h|²`、导频拟合残差都是四者最优，单流 QAM 距离也与能解码的 ue2 同档。**因此下一步应验证数据本身**：核对 `FreqData/data_*` 与本接收机的 DFT 扩频符号顺序/归一化是否一致（而非只看 LLR 幅度——强 LLR 同样可能来自自信的错误判决），并用发射端已知 payload 的独立仿真复现「4 用户各自落在不同波束、ue3 恒 0/6」这一现象。仍建议保存每数据符号、每 UE/天线的频域归一化残差与 LLR 分布，用固定扰码和固定 profile 比较。
2. **可检验的每 comb 对抽头支持选择（已按 §3.1 收窄粒度）。**原提法是「每 UE 独立窗口」，但 §3.1 已证明同一 comb 对内两 UE 用不同窗口会把四个 UE 一起打到 0/4，**选择粒度必须不细于 comb 对**。仍应提出若干物理合理的分数时延、偏移及正则化候选，按完整的相邻 OCC RE 对划分导频训练/留出集（训练矩阵仍须满秩并满足 $|\mathcal P|\ge 2N_{\rm tap}$ 约束），以**留出集预测残差**选窗——in-sample 残差已被证明会选出更差的窗口。记录 $A$ 的条件数、$|\mathcal P|/(2N_{\rm tap})$、留出误差和保持的低码率 CRC。注意按 (pair, 天线) 留出选出的最优窗口彼此差异极大，而单次拟合对四个天线共享同一抽头窗，这是该思路的真实瓶颈。避免通过最终四个 TB CRC 直接搜索数百组参数然后宣称算法泛化；当前 `[-44,42]` 是单抓包调优结果。
3. **CRC 门控的数据辅助 CSI 更新。**只用已经通过本机 TB CRC、且使用正确显式扰码重编码的 UE 符号作为已知分量；未知 UE 仍是干扰而不是训练标签。用独立的符号/子载波留出集检验更新误差，按用户/天线估计泄漏和符号时变，不要在输出上把已抵消用户的信道列置零再调用原 4 流等化器。比较 ue3 每 CB CRC 与其他用户是否退化，而非只看拟合残差。
4. **译码器—检测器迭代（`soft-mmse-pic` 已实现）。**实现区分码块位序的 LDPC posterior-minus-channel 外信息与已使用通道 LLR，并恢复交织、扰码和 PUSCH bit 顺序；外循环输出由最终检测器 LLR 再经 TBDecoder 判定。单帧高码率结果显示 1 轮为 2/4、4 轮为 3/4，ue3 仍失败；配对 CDL 曲线则显示默认 1 轮优于 MMSE-PIC(4) 基线，但约多 35–43 s/点。后续只需在同一配对设置下评估阻尼/轮数的性能与运行成本，不可把后验 LLR 原样反馈或把单帧 CRC 当 BLER 结论。
5. **必要时扩展观测模型。**若残差呈频率/符号相关结构，再评估分数时延、残余 CFO/ICI、非满 CP 多径、独立 UE 定时或 IQ 非理想；用真实独立抓包或有已知发射 bit/信道的仿真验证。不能把参考链路自身 CRC 失败的 `ue*_tx_bits` 当成 oracle 用于调 CSI，也不能通过禁用 CRC、硬编码 UE bit 或只针对该 H5 的索引特判获得表面的 4/4。

若后续要把 `soft-mmse-pic` 宣称为高码率抓包的解码改善，入库条件仍应是 4/4 TB CRC、逐 CB 标志自洽、此前通过的 UE 不回退，并保留固定输入回归；目前 3/4 只能报告为单帧观察。算法自身的采用价值由独立已知 payload 仿真上的 BLER/BER、数值稳定性及计算/显存成本判断。

## 6. CP-OFDM 原生逐 RE 接收路径

本节描述独立于前述 DFT-s-OFDM 抓包优化的仿真路径；这些抓包结论不外推到 CP-OFDM。CP 输入保留 Sionna 原生资源图，按资源元素检测，不做 DFT 解扩/IDFT。支持 LMMSE、LMMSE-SIC、K-best、EP、MMSE-PIC 与 soft-MMSE-PIC；K-best 要求接收天线数不少于所有 UE 的总层数。CP 的 EP 路径使用 double precision：CPU 高 SNR 回环中 float32 EP 未通过 CRC，double precision 通过。代价是更高的计算与内存开销。

DMRS 估计使用原生 pilot mask，而不是把整个 DMRS 符号都当成导频。Type-2 DMRS 可与数据 RE 同处一个 OFDM 符号；`dmrs-lmmse` 只将真实 pilot RE 放入抽头拟合，过滤全零设计行并检查剩余矩阵秩。CP 的 `dmrs_beta` 必须匹配当前原生 Sionna DMRS 配置，配置不兼容时在构造 transmitter 前报错。

回归覆盖 `configs/pusch_cp_2ue_2layer.toml` 的 2 UE × 2 层频域/时域信道路径、perfect/DMRS CSI、六种检测器与 DMRS tap-prior LMMSE；另有 8 流 perfect/DMRS 解码及 type-2 pilot/data 同符号用例。`configs/bler_cp_smoke.toml` 只作两 TB 单帧连通性检查，不代表统计 BLER。端到端 CLI 示例见 `README.md` 的 CP-OFDM smoke。

