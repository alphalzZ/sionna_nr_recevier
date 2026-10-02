# Release 0.1 feature catalog

This catalog describes checked-in behavior, not feature intent. Implementation references use repository-relative paths and line numbers. Test files establish coverage locations; a test/smoke result is not promoted to statistical or hardware-performance evidence.

## Public entry points and artifacts

| Entry point | Input / behavior | Output | Implementation / coverage |
|---|---|---|---|
| `nr-pusch-tx` | TOML TX profile; seeded payload generation and batch sizing | NPZ `iq`, `frequency_grid`, `bits` plus JSON settings/version/axis metadata | `src/nr_pusch/cli/tx.py`; `src/nr_pusch/transmitter.py:17-25,28-135`; `src/nr_pusch/artifacts.py:17-47`; `tests/unit/test_tx_topology_config.py`, `tests/integration/test_pusch_transmitter.py`, `tests/integration/test_mimo_transmitter.py` |
| `nr-pusch-channel` | TX NPZ, CDL TOML; time-IQ or frequency-grid mode | NPZ channel output and JSON sidecar | `src/nr_pusch/cli/channel.py:18-37`; `src/nr_pusch/channel.py:21-39,42-85,176-253`; `tests/unit/test_channel_config.py`, `tests/integration/test_cdl_channel.py`, `tests/integration/test_mimo_channel.py` |
| `nr-pusch-rx` | Capture NPZ or MATLAB H5; RX/TX-compatible TOML; estimator/detector, domain, noise and optional scrambling/CB-CRC settings | NPZ bits, TB CRC, constellation real/imag plus JSON receiver and reference-comparison metadata | `src/nr_pusch/cli/rx.py:21-75,118-196,251-313`; `src/nr_pusch/receiver.py:1532-1537,1554-1606,1790-1857`; `tests/unit/test_matlab_h5.py`, `tests/unit/test_dmrs_lmmse.py`, `tests/integration/test_pusch_receiver.py`, `tests/integration/test_mimo_receiver.py`, `tests/integration/test_web_rx_import.py` |
| `nr-pusch-bler` | TX/CDL/BLER TOMLs; estimator, detector, SNR and batch sweeps | CSV point rows, JSON manifest, optional JSONL progress | `src/nr_pusch/cli/bler.py`; `src/nr_pusch/bler.py:31-58,128-288,326-372`; `src/nr_pusch/simulation_config.py:12-43,84-153`; `tests/unit/test_bler_sweep.py`, `tests/integration/test_mimo_bler.py` |
| `nr-pusch-estimator-validation` | TX/CDL/validation TOMLs; train an exact-setting tap prior and paired DMRS validation | JSON summary, per-frame NPZ, prior NPZ | `src/nr_pusch/cli/estimator_validation.py`; `src/nr_pusch/estimator_validation.py:26-42`; `src/nr_pusch/dmrs_prior.py`; `tests/unit/test_estimator_validation.py`, `tests/integration/test_dmrs_lmmse.py` |
| `nr-pusch-web` | Local dashboard backed by selected TOML and persistent run directories | Browser dashboard; run/config/RX HTTP APIs; run and decode downloads | `src/nr_pusch/cli/web.py`; `src/nr_pusch/web.py:305-331,338-366,692-783`; `src/nr_pusch/web_static/`; `tests/integration/test_web_estimator_matrix.py`, `tests/integration/test_web_rx_import.py` |

TX archive arrays are `complex64` for IQ/grid and `uint8` for bits (`artifacts.py:24-44`). The channel and RX command paths also produce NPZ plus JSON sidecars (`cli/channel.py:87-108`, `cli/rx.py:284-313`). BLER saves CSV+JSON (`bler.py:326-372`). Estimator validation saves the summary and frame/prior archives (`estimator_validation.py`). Web routes expose config list/read/save, run list/create/get/cancel, CSV/JSON results, RX config/default information, and RX decode/NPZ/JSON downloads (`web.py:734-763`).

## Implemented coverage

| Area | Implemented behavior | Evidence and boundary |
|---|---|---|
| TX waveforms and topology | CP-OFDM and DFT-s-OFDM; per-UE PUSCH configs; 1–4 layers per UE; 1/2/4 physical antenna ports; at most 8 aggregate streams; codebook and non-codebook parameters; MCS mapping; per-user DMRS port sets; deterministic generation by seed | `config.py:73-123,125-157,164-204`; `transmitter.py:28-135,137-214`; tests `test_tx_topology_config.py`, `test_pusch_transmitter.py`, `test_mimo_transmitter.py` |
| DMRS/configuration | Config carries mapping type, symbol allocation, DMRS config type, type-A position, additional position, CDM groups, length, beta and DMRS ports. CP-OFDM validates native Sionna beta; receiver supports CP type-2 through native LS and generic OCC/tap fitting where needed | `config.py:32-51,177-202`; `receiver.py:1695-1755`; tests `test_dmrs_lmmse.py`, `test_tx_topology_config.py`, `test_pusch_receiver.py` |
| CDL and array topology | Uplink CDL-A–E; independent UE links summed at BS; configured TX/RX rectangular arrays; time convolution includes channel tail; frequency mode applies finite-tap response per OFDM symbol and retains per-UE channels | `channel_config.py:51-82`; `channel.py:42-85,100-174,176-268`; tests `test_channel_config.py`, `test_cdl_channel.py`, `test_mimo_channel.py` |
| Noise | Seeded complex AWGN on time IQ and resource-grid entrypoints; power/variance is measured per receive antenna (time) or per receive/grid stream (frequency) | `noise.py:10-20,22-68`; `tests/unit/test_noise.py`, `tests/integration/test_mimo_bler.py` |
| RX estimators and decode | Time-IQ and frequency-grid receive paths; perfect CSI, DMRS LS and DMRS-LMMSE prior estimator; native TB decode and TB CRC; optional code-block CRC; DMRS bulk-delay estimate; reported bit/CRC/constellation metadata | `receiver.py:1554-1606,1695-1771,1790-1857,1878-1993`; `cli/rx.py:178-183,284-313,338-343`; tests `test_dmrs_lmmse.py`, `test_pusch_receiver.py`, `test_mimo_receiver.py`, `test_estimator_validation.py` |
| RX detector set | LMMSE, strongest-first CRC-gated LMMSE-SIC, K-best, EP, MMSE-PIC and soft-MMSE-PIC. CP-OFDM invokes native per-RE Sionna detectors for K-best/EP/PIC; DFT-s-OFDM uses the project spread-symbol paths | `receiver.py:793-932,934-1053,1554-1606,1638-1694`; tests `test_soft_mmse_pic.py`, `test_pusch_receiver.py`, `test_mimo_receiver.py`, `test_mimo_bler.py` |
| Simulation and validation | SNR-vs-BLER counts CRC or payload mismatches as TB errors, reports frames/TBs/bit errors/BLER/BER/runtime per point, detector batching, optional JSONL callback, and optional stop-at-zero skip records. Paired estimator validation has development/holdout/high-SNR splits and a separate acceptance gate | `bler.py:31-58,128-288,326-372`; `simulation_config.py:12-153`; `estimator_validation.py:26-42,375-500`; tests `test_bler_sweep.py`, `test_estimator_validation.py`, `test_web_estimator_matrix.py` |
| MATLAB and capture I/O | MATLAB TX/RX H5 readers, optional separated data/pilot arrays and transmitted bits, optional per-UE scrambling arrays, NPZ IQ/grid decode, H5 reference bit/CRC comparison | `iq/matlab_h5.py:12-23,26-124,155-198`; `cli/rx.py:118-150,184-196,259-282`; tests `test_matlab_h5.py`, `test_web_rx_import.py` |
| Web/service | Local configuration dashboard; list/read/save profile; create/poll/cancel/list BLER jobs; progress, logs, CSV/JSON result downloads; receive-decode input and NPZ/JSON downloads | `web.py:305-331,338-366,734-763`; tests `test_web_estimator_matrix.py`, `test_web_rx_import.py`. Source/test coverage is not a browser/GPU release smoke. |

### Shipped TOML profiles

| Profile | Checked-in intent and configuration |
|---|---|
| TX `pusch_4ue.toml` | DFT-s-OFDM, 4 UEs × 1 layer/port, 50 RB, MCS table 1/index 20, type-1 DMRS, 30-kHz SCS. `configs/pusch_4ue.toml`. |
| TX `pusch_1ue_1tx_4rx.toml` | DFT-s-OFDM, 1 UE × 1 layer/port, 50 RB, MCS 8. Pair with 4-RX CDL. |
| TX `pusch_1ue_4layer.toml` | DFT-s-OFDM, 1 UE × 4 layers/ports, 50 RB, MCS 8. Pair with 4-TX/8-RX CDL. |
| TX `pusch_2ue_4layer.toml` | DFT-s-OFDM, 2 UEs × 4 layers/ports each (8 streams), 50 RB, MCS 8, DMRS length 2. Pair with 4-TX/8-RX CDL. |
| TX `pusch_cp_2ue_2layer.toml` | CP-OFDM, 2 UEs × 2 layers/ports each (4 streams), 12 RB, MCS 8, non-codebook. Pair with 2-TX/4-RX CDL. |
| CDL `cdl_38_901_4x4.toml` | CDL-A uplink, 1 TX port per UE, 2×2=4 BS antennas, 3.5 GHz, 100-ns delay spread, zero speed, single polarization. |
| CDL `cdl_38_901_2tx_4rx.toml` | CDL-A uplink, 1×2 TX array, 2×2=4 BS antennas; pair with CP 2UE×2layer. |
| CDL `cdl_38_901_4tx_8rx.toml` | CDL-A uplink, 2×2 TX array, 2×4=8 BS antennas; pair with multilayer TX profiles. |
| RX `rx_pusch_4ue.toml` | Four single-layer DFT-s-OFDM users, frequency-domain input, DMRS estimator, MMSE-PIC(8), damping 0.5, noise variance 0.001, tap window limit 6 μs; native DMRS port ordering. |
| RX `rx_pusch_4ue_mcs27.toml` | Four single-layer DFT-s-OFDM users at MCS 27, frequency input, DMRS/MMSE-PIC(8), `l_min=-44`, noise variance 0.0003, 2-μs spread setting. |
| BLER `bler_smoke.toml` | Small four-user BLER diagnostic; SNR -5/20/40/60 dB, max 2 frames/point, CPU defaults. |
| BLER `bler_8stream_smoke.toml` | One-frame frequency-domain 8-stream LMMSE diagnostic at 65 dB; CPU. |
| BLER `bler_cp_smoke.toml` | One-frame frequency-domain CP-OFDM LMMSE diagnostic at 75 dB; CPU, `l_min=-2`, 0.3-μs spread. |
| BLER `bler_4ue_cdl.toml` | Four-UE CDL simulation; eight SNRs -5…30 dB, 2-frame batches, up to 1000 frames/point, 100 errors, six detectors, CPU. |
| BLER `bler_4ue_cdl_gpu.toml` | Historical GPU profile; seven SNRs 20…50 dB, base batch 20 and soft-PIC batch 8, up to 1000 frames/point, stop-at-zero enabled. Its comments record K-best CUDA OOM at batch 20; not the new baseline. |
| BLER `bler_4ue_cdl_validation.toml` | 8-frame/point three-SNR soft-MMSE-PIC estimator diagnostic; CPU. |
| BLER `bler_4ue_cdl_validation_offsets.toml` | Same diagnostic with delay estimation enabled. |
| BLER `bler_estimator_matrix.toml` | Six-SNR soft-MMSE-PIC matrix for DMRS/DMRS-LMMSE/perfect; requires configured tap prior; CUDA. |
| Validation `channel_estimation_validation.toml` | 256 prior realizations, 512 development frames/SNR, 3000 holdout frames/SNR, 10,000 bootstrap replicates, 60-dB/64-frame high-SNR check; CPU as checked in. |

Profile parameters are snapshot-specific; the table does not claim the profiles were run in this catalog pass. Source snapshots are in `configs/` and the paired README/profile comments.

## Explicit limits and interpretation

- TX supports at most 8 aggregate streams, 1–4 layers per UE and 1/2/4 antenna ports; SCS is 15/30/60/120/240 kHz (`config.py:73-103`).
- DFT-s-OFDM accepts type-1 DMRS only, exactly two CDM groups, additional-position values 0/1/2 and a 2/3/5-smooth BWP DFT length (`config.py:104-123`). DMRS length is configured but the specialized low-complexity estimator is selected only for exactly four users, one layer/port, non-codebook and DMRS length 1; other supported cases take the generic rank/OCC-checked estimator (`receiver.py:1713-1755`). Group/sequence hopping fields are not exposed by this TOML model.
- DMRS-LMMSE requires a positive tap-power prior; BLER/RX loaders compare its recorded TX/CDL, tap window, FFT and sample-rate compatibility before use (`receiver.py:1583-1588`; `bler.py:156-172`; `cli/rx.py:203-235`). Priors are not portable across changed waveform/channel/window combinations.
- Perfect CSI requires simulated CDL truth (time taps or frequency response) and is an upper bound, not an H5 mode. H5 data is frequency-domain; the H5 RX path rejects perfect CSI and constrains input profiles to four users, one layer and one TX antenna (`receiver.py:1802-1810,1842-1855`; `cli/rx.py:124-127,192-196`).
- H5 reader expects four RX antennas and the `FreqData/IQdataPdu_real/imag` dataset pair; optional data/pilot/bits are only available when the matching datasets exist (`iq/matlab_h5.py:37-99`). Explicit scrambling input is separate and requires one binary sequence per UE (`matlab_h5.py:103-124`).
- Custom DMRS estimators require one BS receiver endpoint (`receiver.py:1471-1475`). Frequency-domain CDL assumes sufficient cyclic prefix and excludes ISI; time-domain processing models sample-level convolution and channel tail (`channel.py:78-85,176-246`). Only CDL A–E and single-polarized arrays are accepted (`channel_config.py:51-72`).
- K-best dimensional cost rises with stream count; use 8-stream configs only for bounded correctness smokes, not exponential K-best/EP performance sweeps. The shipped GPU profile notes K-best OOM for batch 20 (`configs/bler_4ue_cdl_gpu.toml:3-8`).
- K-best requires physical RX antennas at least equal to aggregate streams in both waveform implementations (`receiver.py:283-285,888-890`); DFT-s-OFDM K-best additionally requires one BS endpoint (`receiver.py:283-285`).
- Delay estimation reports a diagnostic bulk delay; it is not capture time alignment. BLER's optional zero-error stop skips all later configured SNR entries after the first zero-error point, so ordered increasing SNR is an assumption (`bler.py:137-141,274-287`).
- BLER point `runtime_s` covers point work, while model setup and export occur outside it; process wall time must be recorded separately (`bler.py:149-188,197-268,326-372`). Device memory sampled by system telemetry is total GPU use, not PyTorch allocator peak.
- Checked-in unit/integration tests are correctness/control evidence only. Low-frame smoke BLER and reduced paired validation are diagnostic, not statistical release acceptance or performance claims.

## Test map

- Config/topology/noise/device/import: `tests/unit/test_tx_topology_config.py`, `test_channel_config.py`, `test_noise.py`, `test_device.py`, `test_matlab_h5.py`, `test_simulation_config.py`.
- Simulation/detector/prior: `tests/unit/test_bler_sweep.py`, `test_dmrs_lmmse.py`, `test_soft_mmse_pic.py`, `test_estimator_validation.py`.
- Sionna end-to-end: `tests/integration/test_pusch_transmitter.py`, `test_mimo_transmitter.py`, `test_cdl_channel.py`, `test_mimo_channel.py`, `test_pusch_receiver.py`, `test_mimo_receiver.py`, `test_dmrs_lmmse.py`, `test_mimo_bler.py`.
- Web API/RX import and matrix: `tests/integration/test_web_rx_import.py`, `test_web_estimator_matrix.py`.

The full standard-library unittest discovery command is the release correctness gate; these file locations alone do not assert that it passed.
