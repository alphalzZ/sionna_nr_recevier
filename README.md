# NR PUSCH Lab

一个小型、配置驱动的 5G NR PUSCH 发送与 IQ 分析仓库。首批目标为 4 用户上行 MU-MIMO，每用户 1 层；发送端通过独立 IQ 文件和运行清单与接收端解耦。

## 当前发送端

当前实现以 Sionna 2.0.1 的 `PUSCHTransmitter` 生成 PUSCH 数据资源网格，并支持 CP-OFDM 和 DFT-s-OFDM 波形输出。DFT-s-OFDM 对数据符号应用酉 DFT，并按 type-1 低 PAPR 序列生成 DMRS 后进行 OFDM 调制。由于 Sionna 2.0.1 的组合 PUSCH 发射器不支持 transform precoding，配置中的 DFT 预编码 MCS 会映射到同调制阶数和码率的原生 Sionna MCS，以复用其 TB 编码器。当前配置为 DFT-s-OFDM MCS table 1/index 20，目标码率 0.6015625，TB 大小 28,168 bit。

## 运行

使用仓库指定的虚拟环境安装本仓库：

```bash
/home/le-lei/workspace/test/.venv/bin/pip install -e .
nr-pusch-tx --config configs/pusch_4ue.toml --output runs/first_tx.npz --batch-size 1 --seed 7
```

NPZ 中 `iq` 的轴顺序为 `[batch, user, tx_antenna, sample]`，`frequency_grid` 为 Sionna PUSCH CP-OFDM 资源网格（`[batch, user, tx_antenna, symbol, subcarrier]`），`bits` 为 `[batch, user, transport_block_bit]`。同名 JSON manifest 保存已解析的运行配置、轴含义和依赖版本。

## 目录

- `configs/`：发送与抓包分析配置
- `src/nr_pusch/`：配置解析、Sionna 适配、IQ 导出和命令入口
- `tests/fixtures/`：后续放置 MATLAB H5 参考夹具
- `docs/`：支持范围、接口和夹具说明
