# 四用户 DFT-s-OFDM PUSCH 接收机：算法设计与优化记录

本文记录仓库**现有实现**、对四组 MATLAB 接收抓包的**实测结果**、已经试过但未保留的方案，以及后续实验的判据。重点是高码率 `RxTestVectorCase123427.h5`；文中的 4/4 均指**四个 UE 的本机 TB CRC 全部通过**，不是与 H5 中参考载荷逐 bit 相同。实现入口为 `src/nr_pusch/receiver.py`，运行参数分别在 `configs/rx_pusch_4ue.toml` 和 `configs/rx_pusch_4ue_mcs27.toml`。

## 1. 结论与适用边界

- 当前已保留的改进是**调整高码率抓包的 DMRS 有限抽头拟合窗口**，不是一个已完成的自适应波束检测器：窗口从默认下界 `l_min=-6` 移至 `-44`，`max_delay_spread_s=2e-6`，保持 MMSE-PIC、8 次迭代、阻尼 0.5、接收噪声方差 0.0003。该抓包由 **2/4 提升到 3/4 TB CRC**；ue2 的码块从 0/6 提升到 6/6，ue3 仍为 0/6。**4/4 尚未实现**。
- 低码率抓包统一沿用 `rx_pusch_4ue.toml`，没有把高码率抓包专用的窗口强加给其他抓包：`RxTestVector.h5` 为 4/4，`Case11121314` 为 4/4，`Case78914` 为 3/4（ue0 未通过）。
- 本接收机目前假定 4 UE、每 UE 1 层、4 根接收天线、一个 DMRS OFDM 符号、两组各两个 UE 的共享 comb/OCC、配置对应的完整 DFT-s-OFDM 数据符号。只有这些几何条件通过代码检查；不代表支持任意 NR PUSCH 配置。
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

代码以 `torch.linalg.lstsq` **联合**求解这两个 UE 在每根 RX 天线上的抽头，用 $B$ 合成全部有效子载波的频响，并把单个 DMRS 符号的估计广播到同一 slot 的数据符号。用 $\operatorname{pinv}(A^HA)$ 的用户块、基函数与输入噪声方差计算近似的逐频点 CSI 误差方差。每组导频 RE 数必须至少覆盖两个 UE 合计的抽头数：代码约束为 $|\mathcal P|\ge2N_{\rm tap}$；窗口越宽，拟合自由度与噪声放大风险也越高。该误差方差只建模 LS 噪声传播，不保证覆盖 ICI、模型外多径或未建模的符号内变化。

`l_max` 使用 Sionna 的 `time_lag_discrete_time_channel(sample_rate_hz,max_delay_spread_s)` 确定；在本抓包 18 MHz 下，2 µs 对应 `l_max=42`，所以高码率 profile 的拟合窗口是 **[-44,42]，共 87 个抽头/UE**。默认 `l_min=-6`、3 µs 则是 [-6,60]。改变 `l_min` 不等于事后给现有 CSI 乘一条相位斜坡：它直接改变了**联合拟合的列空间**，会改变两个共享 comb 用户的分离、插值偏差和噪声传播。这是此次 ue2 恢复的可重复配置变化；无法仅凭 CRC 将原因唯一归结为某一条物理路径。

可选的 `--estimate-delay` 是输出诊断：从频响 FFT 峰值和抛物线插值计算接收天线的 bulk offset，**不修正 CSI**。当前实现按用户循环时会覆盖 `last_offsets`，sidecar 最终只记录**最后处理的用户**在各天线上的估计，不是完整的四用户时延表。可选 `--spatial-denoise` 把每用户跨天线频响投影到宽带空间协方差的主特征向量，等价于强加近似秩一空间签名；这两个开关默认都关闭。

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
| 延迟补偿、空间秩一去噪、CPE（未保留或默认关） | `--estimate-delay` 只输出峰值诊断不改 CSI；直接把估计 offset 作为相位斜坡补偿曾使低码率正常抓包从 4/4 降到 1/4。`--spatial-denoise` 的秩一投影使两个原 4/4 抓包降为 3/4。盲四阶矩/判决引导 CPE 原型也曾使 4/4 降为 3/4，并出现非有限值。 | 不要仅凭某一接收天线主径就假定每 UE 4 天线信道严格秩一；相位补偿需与拟合窗/分数时延模型一致。CPE 原型未合入。 |
| LDPC / CRC 辅助探索（未保留） | 在混合 CSI 的 ue3 CB0 上尝试不同 LDPC 校验节点更新（boxplus、minsum 等）、flooding/layered、10–160 轮，CB0 仍未通过；对低置信 bit 的小规模 CRC 列表翻转，无候选通过最终 TB CRC。 | 先提高 CB0 检测软信息可信度；提高 LDPC 轮数、仅调整 LLR 尺度或凭 CB CRC 匹配并不能保证 TB 正确。 |

上表比较的是固定 H5 抓包的**同一组接收样本**；改变 `noise_variance` 改的是接收机的噪声假设，不是在 H5 上重新加噪。另一方面，跨独立仿真运行的 BLER 结果不一定使用逐帧相同的随机信道和噪声；如要做配对比较，须控制完整扫描配置、随机种子和 batch 分组。`README.md` 中的历史 BLER 曲线不应直接解释为某一修改在这个 H5 上的因果证据。

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

回归基线：`tests/integration/test_matlab_rx_capture.py` 检查原始抓包全 4 UE 解码、高码率 profile 至少 ue0–ue2 CRC 通过、SIC 内部使用显式扰码进行 CRC 门控；单帧测试**不证明**新算法对真实信道的 BLER 改善。最近一次完整测试为 38 项通过；低码率三组 CLI 抓包分别测得 4/4、4/4、3/4。

## 5. 下一轮优化设计与实验纪律

1. **先定位 ue3 CB0 的误差来源，而非只调最终 CRC。**保存每数据 OFDM 符号、每 UE/天线的 DMRS 拟合残差、频域归一化残差、有效噪声和 LLR 分布；画出 CB0 相关 RE 与其它五个 CB 的可靠性差异。用固定扰码、固定 profile 和相同抓包比较，区分“偶发软信息过度自信”与“整段 CSI 偏差”。只有一个 DMRS 符号时，不能声称从导频直接测得整 slot 的时变信道。
2. **可检验的每 UE/天线抽头支持选择。**提出若干物理合理的分数时延、偏移及正则化候选，按完整的相邻 OCC RE 对划分导频训练/留出集（训练矩阵仍须满秩并满足 RE 数约束），以留出集预测残差或可重复的可靠性准则选窗；记录 $A$ 的条件数、$|\mathcal P|/(2N_{\rm tap})$、拟合误差和保持的低码率 CRC。避免通过最终四个 TB CRC 直接搜索数百组参数然后宣称算法泛化；当前 `[-44,42]` 是单抓包调优结果。
3. **CRC 门控的数据辅助 CSI 更新。**只用已经通过本机 TB CRC、且使用正确显式扰码重编码的 UE 符号作为已知分量；未知 UE 仍是干扰而不是训练标签。用独立的符号/子载波留出集检验更新误差，按用户/天线估计泄漏和符号时变，不要在输出上把已抵消用户的信道列置零再调用原 4 流等化器。比较 ue3 每 CB CRC 与其他用户是否退化，而非只看拟合残差。
4. **真正的译码器—检测器迭代。**若尝试把 LDPC 软输出送回均衡器，必须区分码字的**外信息**与已被使用的通道 LLR，遵守编码 bit 交织、扰码和 DFT-s-OFDM 符号顺序，防止反复使用同一证据造成虚假置信度。每轮保持逐 CB/TB CRC 记录、有限 LLR 检查，并在无改善时停止。仅将译码器后验 LLR 原样喂回 PIC 不构成可靠的 Turbo 接收机。
5. **必要时扩展观测模型。**若残差呈频率/符号相关结构，再评估分数时延、残余 CFO/ICI、非满 CP 多径、独立 UE 定时或 IQ 非理想；用真实独立抓包或有已知发射 bit/信道的仿真验证。不能把参考链路自身 CRC 失败的 `ue*_tx_bits` 当成 oracle 用于调 CSI，也不能通过禁用 CRC、硬编码 UE bit 或只针对该 H5 的索引特判获得表面的 4/4。

新方案的**入库条件**：在高码率抓包上真实达到 4/4 TB CRC、逐 CB 标志自洽，四组抓包中此前已通过的 UE 不回退；保留固定输入的可复现回归；在已知发送 payload 的独立仿真上比较 BLER/BER、数值稳定性及计算/显存成本。若仅达到 3/4 或 5/6 CB，应明确记录为诊断信息而非“解码完成”。
