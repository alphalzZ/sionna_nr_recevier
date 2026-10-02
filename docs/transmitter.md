# Transmitter interface and MIMO coverage

`TxSettings` builds one native Sionna 2.0.1 `PUSCHConfig` per UE. Users share
PUSCH resource/MCS/layer settings and carry separate transport blocks and
non-overlapping DMRS port sets. Each UE supports 1–4 layers, with at most eight
concurrent layers across all UEs. Native antenna-port counts are 1, 2, or 4;
non-codebook requires one physical antenna per layer. Codebook configurations
may use more antenna ports than layers, with a native TPMI. Sionna validates
the remaining PUSCH/DMRS combinations.

The exported NPZ keeps `iq` as `[batch,user,tx_antenna,sample]`,
`frequency_grid` as `[batch,user,tx_antenna,symbol,subcarrier]`, and `bits` as
`[batch,user,tb_bit]`. The antenna axis is physical after optional codebook
precoding; it is *not* the layer axis. The JSON manifest records all three axis
definitions, users, layers per UE, physical antennas per UE, and total streams.

CP-OFDM uses Sionna's native pilot, layer mapping, and precoder. For
DFT-s-OFDM the adapter applies unitary DFT per data-bearing layer, maps
type-1 low-PAPR DMRS, and uses native DMRS symbol positions, time OCC,
and codebook precoding. The MATLAB TX reference uses comb-first port order
(ports 0/2 on the even comb); the supplied MATLAB RX captures use native
OCC-first order (ports 0/1 on the even comb). RX capture profiles explicitly
set `dft_s_dmrs_port_order = "native"`; other profiles default to `"comb-first"`.
Single-symbol DMRS permits type-1 ports 0–3; eight streams require length 2,
ports 0–7, and `dmrs_additional_position` 0 or 1. This adapter supports no
group/sequence hopping. Sionna's composite lacks transform-precoding MCS
tables, so the adapter selects an equivalent native MCS with the same
modulation order and target rate, retaining the configured MCS in metadata.
