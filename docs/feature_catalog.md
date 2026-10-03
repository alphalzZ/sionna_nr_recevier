# Release 0.1 feature catalog

This catalog describes checked-in behavior, not feature intent. Implementation references use repository-relative paths and line numbers. Test files establish coverage locations; a test/smoke result is not promoted to statistical or hardware-performance evidence.

## Public entry points and artifacts

| Entry point | Input / behavior | Output | Implementation / coverage |
|---|---|---|---|
| `nr-pusch-tx` | TOML TX profile; seeded payload generation and batch sizing | NPZ `iq`, `frequency_grid`, `bits` plus JSON settings/version/axis metadata | `src/nr_pusch/cli/tx.py`; `src/nr_pusch/transmitter.py:17-25,28-135`; `src/nr_pusch/artifacts.py:17-47`; `tests/unit/test_tx_topology_config.py`, `tests/integration/test_pusch_transmitter.py`, `tests/integration/test_mimo_transmitter.py` |
| `nr-pusch-channel` | TX NPZ, CDL TOML; time-IQ or frequency-grid mode | NPZ channel output and JSON sidecar | `src/nr_pusch/cli/channel.py:18-37`; `src/nr_pusch/channel.py:21-39,42-85,176-253`; `tests/unit/test_channel_config.py`, `tests/integration/test_cdl_channel.py`, `tests/integration/test_mimo_channel.py` |
| `nr-pusch-rx` | Capture NPZ or MATLAB H5; RX/TX-compatible TOML; estimator/detector, domain, noise and optional scrambling/CB-CRC settings | NPZ bits, TB CRC, constellation real/imag plus JSON receiver and reference-comparison metadata | `src/nr_pusch/cli/rx.py:21-75,118-196,251-313`; `src/nr_pusch/receiver.py:1532-1537,1554-1606,1790-1857`; `tests/unit/test_matlab_h5.py`, `tests/unit/test_dmrs_lmmse.py`, `tests/integration/test_pusch_receiver.py`, `tests/integration/test_mimo_receiver.py`, `tests/integration/test_web_rx_import.py` |
| `nr-pusch-bler` | TX/CDL/BLER TOMLs; estimator, detector, SNR and batch sweeps | CSV point rows, JSON manifest, optional JSONL progress | `src/nr_pusch/cli/bler.py`; `src/nr_pusch/bler.py:31-66,128-288,326-375`; `src/nr_pusch/simulation_config.py:12-43,84-153`; `tests/unit/test_bler_sweep.py`, `tests/integration/test_mimo_bler.py` |
| `nr-pusch-estimator-validation` | TX/CDL/validation TOMLs; train an exact-setting tap prior, paired DMRS validation and acceptance-gated publication into the shared prior directory | JSON summary, per-frame NPZ, diagnostic prior NPZ, published prior | `src/nr_pusch/cli/estimator_validation.py`; `src/nr_pusch/estimator_validation.py:30-45`; `src/nr_pusch/dmrs_prior.py`; `tests/unit/test_estimator_validation.py`, `tests/integration/test_dmrs_lmmse.py` |
| `nr-pusch-web` | Local dashboard backed by `configs/scenarios.toml` curated TX/CDL/BLER bundles; advanced independent TOML editing with combination preflight before queueing | Browser dashboard; run/config/RX HTTP APIs; run and decode downloads | `src/nr_pusch/cli/web.py`; `src/nr_pusch/web.py:120-340,482-590,997-1008`; `src/nr_pusch/web_static/`; `tests/integration/test_web_estimator_matrix.py`, `tests/integration/test_web_rx_import.py` |

TX archives store IQ/grid as complex64 and bits as uint8 (`artifacts.py:24-44`). Channel and RX commands also produce NPZ plus JSON sidecars (`cli/channel.py:87-108`, `cli/rx.py:284-313`). BLER saves CSV+JSON (`bler.py:326-375`). Estimator validation saves the summary and frame/prior archives (`estimator_validation.py`). Web routes expose config list/read/save, run list/create/get/cancel, CSV/JSON results, RX config/default information, and RX decode/NPZ/JSON downloads (`web.py:997-1008`).

## Implemented coverage

| Area | Implemented behavior | Evidence and boundary |
|---|---|---|
| TX waveforms and topology | CP-OFDM and DFT-s-OFDM; per-UE PUSCH configs; 1–4 layers per UE; 1/2/4 physical antenna ports; at most 8 aggregate streams; codebook and non-codebook parameters; MCS mapping; per-user DMRS port sets; deterministic generation by seed | `config.py:73-123,125-157,164-204`; `transmitter.py:28-135,137-214`; tests `test_tx_topology_config.py`, `test_pusch_transmitter.py`, `test_mimo_transmitter.py` |
| DMRS/configuration | Config carries mapping type, symbol allocation, DMRS config type, type-A position, additional position, CDM groups, length, beta and DMRS ports. CP-OFDM validates native Sionna beta; receiver supports CP type-2 through native LS and generic OCC/tap fitting where needed | `config.py:32-51,177-202`; `receiver.py:1695-1755`; tests `test_dmrs_lmmse.py`, `test_tx_topology_config.py`, `test_pusch_receiver.py` |
| CDL and array topology | Uplink CDL-A–E; independent UE links summed at BS; configured TX/RX rectangular arrays; time convolution includes channel tail; frequency mode applies finite-tap response per OFDM symbol and retains per-UE channels | `channel_config.py:51-82`; `channel.py:42-85,100-174,176-268`; tests `test_channel_config.py`, `test_cdl_channel.py`, `test_mimo_channel.py` |
| Noise | Seeded complex AWGN on time IQ and resource-grid entrypoints; power/variance is measured per receive antenna (time) or per receive/grid stream (frequency) | `noise.py:10-20,22-68`; `tests/unit/test_noise.py`, `tests/integration/test_mimo_bler.py` |
| RX estimators and decode | Time-IQ and frequency-grid receive paths; perfect CSI, DMRS LS and DMRS-LMMSE prior estimator resolved from the shared prior directory by channel configuration; native TB decode and TB CRC; optional code-block CRC; DMRS bulk-delay estimate; reported bit/CRC/constellation metadata | `receiver.py:1554-1606,1695-1771,1790-1857,1878-1993`; `cli/rx.py:44-54,206-245`; tests `test_dmrs_lmmse.py`, `test_pusch_receiver.py`, `test_mimo_receiver.py`, `test_estimator_validation.py` |
| RX detector set | LMMSE, strongest-first CRC-gated LMMSE-SIC, K-best, EP, MMSE-PIC and soft-MMSE-PIC. CP-OFDM invokes native per-RE Sionna detectors; CP K-best keeps its Cholesky fast path and falls back to QR for singular channel Gram matrices. DFT-s-OFDM uses project spread-symbol paths | `receiver.py:799-1004,1554-1606,1638-1694`; tests `test_soft_mmse_pic.py`, `test_pusch_receiver.py`, `test_mimo_receiver.py`, `test_mimo_bler.py` |
| Simulation and validation | SNR-vs-BLER counts CRC or payload mismatches as TB errors, reports frames/TBs/bit errors/BLER/BER/runtime per point, detector batching, optional JSONL callback, and optional stop-at-zero skip records. Paired estimator validation has development/holdout/high-SNR splits and a separate acceptance gate | `bler.py:31-66,128-288,326-375`; `simulation_config.py:12-153`; `estimator_validation.py:26-42,375-500`; tests `test_bler_sweep.py`, `test_estimator_validation.py`, `test_web_estimator_matrix.py` |
| MATLAB and capture I/O | MATLAB TX/RX H5 readers, optional separated data/pilot arrays and transmitted bits, optional per-UE scrambling arrays, NPZ IQ/grid decode, H5 reference bit/CRC comparison | `iq/matlab_h5.py:12-23,26-124,155-198`; `cli/rx.py:118-150,184-196,259-282`; tests `test_matlab_h5.py`, `test_web_rx_import.py` |
| Web/service | Curated scenario bundles; advanced independent profile/TOML editing; preflight checks TX/CDL antenna topology, K-best RX stream count and exact DMRS-LMMSE prior resolution from the shared prior directory before queueing; RX decode exposes a DMRS/DMRS-LMMSE estimator and CDL selector; run progress/results and receive-decode workflows | `web.py:120-340,482-590,997-1008`; `web_static/index.html`, `web_static/app.js`; tests `test_web_estimator_matrix.py`, `test_web_rx_import.py`. Browser smoke is separate from source/test coverage. |

### Shipped TOML profiles

| Profile | Checked-in intent and configuration |
|---|---|
| TX `pusch_4ue.toml` | DFT-s-OFDM, 4 UEs × 1 layer/port, 50 RB, MCS table 1/index 20, type-1 DMRS, 30-kHz SCS. `configs/pusch_4ue.toml`. |
| TX `pusch_1ue_1tx_4rx.toml` | DFT-s-OFDM, 1 UE × 1 layer/port, 50 RB, MCS 8. Pair with 4-RX CDL. |
| TX `pusch_1ue_4layer.toml` | DFT-s-OFDM, 1 UE × 4 layers/ports, 50 RB, MCS 8. Pair with 4-TX/8-RX CDL. |
| TX `pusch_2ue_4layer.toml` | DFT-s-OFDM, 2 UEs × 4 layers/ports each (8 streams), 50 RB, MCS 8, DMRS length 2. Pair with 4-TX/8-RX CDL. |
| TX `pusch_cp_2ue_2layer.toml` | CP-OFDM, 2 UEs × 2 layers/ports each (4 streams), 12 RB, MCS 8, non-codebook. Pair with 2-TX/4-RX CDL. |
| TX `pusch_4ue_4tx_codebook.toml` | Four single-layer DFT-s-OFDM users, 50 RB, 4 antenna ports with codebook precoding. |
| TX `pusch_4ue_4tx_codebook_cp12.toml` | Four single-layer CP-OFDM users, 12 RB, 4 antenna ports with codebook precoding. |
| TX `pusch_cp_2ue_2layer_type2.toml` | Two-user/two-layer CP-OFDM (4 streams), DMRS type 2, length 1 and additional position 1. |
| CDL `cdl_38_901_4x4.toml` | CDL-A uplink, 1 TX port per UE, 2×2=4 BS antennas, 3.5 GHz, 100-ns delay spread, zero speed, single polarization. |
| CDL `cdl_38_901_2tx_4rx.toml` | CDL-A uplink, 1×2 TX array, 2×2=4 BS antennas; pair with CP 2UE×2layer. |
| CDL `cdl_38_901_4tx_8rx.toml` | CDL-A uplink, 2×2 TX array, 2×4=8 BS antennas; pair with multilayer TX profiles. |
| CDL variants `cdl_38_901_4x4_{B,C,D,E}.toml`, `cdl_38_901_2tx_4rx_{B,C,D,E}.toml`, `cdl_38_901_4tx_8rx_{B,C,D,E}.toml` | CDL models B–E at the matching 4×4 UE/BS, 2TX/4RX and 4TX/8RX topologies. |
| CDL `cdl_38_901_4x4_speed.toml` | Four single-antenna UE links, 4-RX BS, CDL-A at 3 m/s. |
| RX `rx_pusch_4ue.toml` | Four single-layer DFT-s-OFDM users, frequency-domain input, DMRS estimator, MMSE-PIC(8), damping 0.5, noise variance 0.001, tap window limit 6 μs; native DMRS port ordering. |
| RX `rx_pusch_4ue_mcs27.toml` | Four single-layer DFT-s-OFDM users at MCS 27, frequency input, DMRS/MMSE-PIC(8), `l_min=-44`, noise variance 0.0003, 2-μs spread setting. |
| BLER `bler_smoke.toml` | Small four-user BLER diagnostic; SNR -5/20/40/60 dB, max 2 frames/point, CPU defaults. |
| BLER `bler_8stream_smoke.toml` | One-frame frequency-domain 8-stream LMMSE diagnostic at 65 dB; CPU. |
| BLER `bler_cp_smoke.toml` | One-frame frequency-domain CP-OFDM LMMSE diagnostic at 75 dB; CPU, tap window follows the paired CDL channel. |
| BLER `bler_8stream_mmse_pic_smoke.toml` | One-frame 8-stream MMSE-PIC(4) diagnostic at 65 dB; CUDA. |
| BLER `bler_dft_4ue_dmrs_lmmse_smoke.toml`, `bler_cp_2ue_2layer_dmrs_lmmse_smoke.toml` | One-frame DMRS-LMMSE detector-matrix smokes at 25 dB; CUDA; resolve their own published prior from the shared prior directory keyed by TX/CDL. |
| BLER `bler_release_smoke.toml` | One-frame 25-dB LMMSE GPU profile used by web smoke. |
| BLER `bler_4ue_cdl.toml` | Four-UE CDL simulation; eight SNRs -5…30 dB, 2-frame batches, up to 1000 frames/point, 100 errors, six detectors, CPU. |
| BLER `bler_4ue_cdl_gpu.toml` | Historical GPU profile; seven SNRs 20…50 dB, base batch 20 and soft-PIC batch 8, up to 1000 frames/point, stop-at-zero enabled. Its comments record K-best CUDA OOM at batch 20; not the new baseline. |
| BLER `bler_4ue_cdl_validation.toml` | 8-frame/point three-SNR soft-MMSE-PIC estimator diagnostic; CPU. |
| BLER `bler_4ue_cdl_validation_offsets.toml` | Same diagnostic with delay estimation enabled. |
| BLER `bler_estimator_matrix.toml` | Six-SNR soft-MMSE-PIC matrix for DMRS/DMRS-LMMSE/perfect; the DMRS-LMMSE arm needs a matching published prior; CUDA. |
| Validation `channel_estimation_validation.toml` | 256 prior realizations, 512 development frames/SNR, 3000 holdout frames/SNR, 10,000 bootstrap replicates, 60-dB/64-frame high-SNR check; CPU as checked in. |
| Validation `channel_estimation_validation_v2.toml` | Same training/holdout/bootstrap settings as version 1 but runs on GPU with `batch_size = 20` and relaxes the acceptance thresholds (`min_relative_bler_reduction = 0.09`, `require_strict_upper_bound = false`); the different batch size resamples channel and AWGN realizations, so its numbers are an independent draw, and the published CP prior's acceptance marker records the v2 thresholds. |

Shipped-profile parameters are snapshot-specific; the table above does not claim every profile was run in this catalog pass. Source profiles live in `configs/`. DMRS-LMMSE profiles carry no prior path: they need a published prior under the ignored local `configs/tap_power_prior/` and fail closed otherwise, so they are not portable to a clean checkout.

### Web scenario bundles

| ID | TX / CDL / BLER profile | Intended use |
|---|---|---|
| `dft-4ue-cdl-a` | `pusch_4ue.toml` / `cdl_38_901_4x4.toml` / `bler_smoke.toml` | 4-user DFT-s-OFDM, 4-RX CDL-A diagnostic |
| `cp-2ue-2layer-cdl-a` | `pusch_cp_2ue_2layer.toml` / `cdl_38_901_2tx_4rx.toml` / `bler_cp_smoke.toml` | 4-stream CP-OFDM, 2-TX/4-RX CDL-A diagnostic |
| `dft-8stream-cdl-a` | `pusch_2ue_4layer.toml` / `cdl_38_901_4tx_8rx.toml` / `bler_8stream_smoke.toml` | Bounded 8-stream LMMSE diagnostic |

## Explicit limits and interpretation

- TX supports at most 8 aggregate streams, 1–4 layers per UE and 1/2/4 antenna ports; SCS is 15/30/60/120/240 kHz (`config.py:73-103`).
- DFT-s-OFDM accepts type-1 DMRS only, exactly two CDM groups, additional-position values 0/1/2 and a 2/3/5-smooth BWP DFT length (`config.py:104-123`). DMRS length is configured but the specialized low-complexity estimator is selected only for exactly four users, one layer/port, non-codebook and DMRS length 1; other supported cases take the generic rank/OCC-checked estimator (`receiver.py:1713-1755`). Group/sequence hopping fields are not exposed by this TOML model.
- DMRS-LMMSE requires a positive tap-power prior; BLER/RX/Web loaders resolve it by exact compatibility key (waveform, TX/CDL geometry, tap window, FFT, sample rate) under the shared prior directory and still compare the recorded metadata before use (`dmrs_prior.py`; `bler.py:148-176`; `cli/rx.py:208-245`; `web.py`). Only priors published after a passing acceptance gate carry the required acceptance marker, so diagnostic candidates are never selected. MCS table/index are excluded because they affect transport-block coding, not channel tap calibration; other recorded geometry or channel mismatches resolve to a different key and report a missing prior. Covered by `tests/unit/test_dmrs_lmmse.py` and `tests/integration/test_web_estimator_matrix.py`.
- `docs/dmrs_tap_power_prior.md` records the tap-power prior theory: which inputs determine it (channel ensemble and the delay-grid mapping), which do not (waveform, MCS, user count, detector), why antenna count only changes averaging precision when all elements share identical statistics and power normalization, the resulting count rule (one prior per channel ensemble × FFT/sample-rate × tap window), the training method, the pre-registered acceptance gate and publication rules, and the current gap: the registry key still hashes the full TX/CDL antenna geometry, so cross-array reuse needs validation before the key can be relaxed.
- `docs/dmrs_tap_power_prior.md` §8.1 records the 2026-10-03 cross-array reuse check on the published DFT-s-OFDM prior: a prior trained on a 1×1 receive array, applied to the 2×2 target, matched the array-matched prior on a 300-frame-per-SNR holdout (paired block-error difference 97.5% upper bound 0.000000, data-RE NMSE ratio 1.0013 / 1.0009, both arms meeting the version-1 criteria on that reduced run, which is exploratory and not the 3000-frame acceptance gate); the two vectors differ by 5.3% relative L2 with a systematic ~3.3% level shift toward the smaller array. The direction measured is 1Rx-trained → 4Rx target, the reverse of the §4 "train on the largest array, share to smaller ones" recommendation, and the 1Rx target with four users cannot decode at all (1200/1200 block errors even with its own prior), so the reverse direction stays unmeasured. The registry key is unchanged.
- Perfect CSI requires simulated CDL truth (time taps or frequency response) and is an upper bound, not an H5 mode. H5 data is frequency-domain; the H5 RX path rejects perfect CSI and constrains input profiles to four users, one layer and one TX antenna (`receiver.py:1802-1810,1842-1855`; `cli/rx.py:124-127,192-196`).
- H5 reader expects four RX antennas and the `FreqData/IQdataPdu_real/imag` dataset pair; optional data/pilot/bits are only available when the matching datasets exist (`iq/matlab_h5.py:37-99`). Explicit scrambling input is separate and requires one binary sequence per UE (`matlab_h5.py:103-124`).
- Custom DMRS estimators require one BS receiver endpoint (`receiver.py:1471-1475`). Frequency-domain CDL assumes sufficient cyclic prefix and excludes ISI; time-domain processing models sample-level convolution and channel tail (`channel.py:78-85,176-246`). Only CDL A–E and single-polarized arrays are accepted (`channel_config.py:51-72`).
- The receiver's DMRS tap window (`l_min`, `max_delay_spread_s`) is a delay prior, not a measurement. The frequency-domain CDL builds its taps inside `time_lag_discrete_time_channel(sample_rate_hz, max_delay_spread_s)`, so a receiver window narrower than the channel's own support truncates the joint fit. Measured on `pusch_cp_2ue_2layer.toml` + `cdl_38_901_2tx_4rx.toml` (4.32 MHz, 26-tap support): a 12-tap `l_min=-2` window left ~9% relative CSI error, and the ~5-tap window implied by `max_delay_spread_s=0.3e-6` failed TB CRC for all six detectors with both `dmrs` and `dmrs-lmmse` while perfect CSI decoded cleanly. BLER sweeps fall back to the channel value when unset (`channel.py:105-106`; `bler.py:151-155`; `simulation_config.py:31-34`); covered by `tests/integration/test_mimo_receiver.py` and `tests/integration/test_mimo_bler.py`.
- K-best dimensional cost rises with stream count; use 8-stream configs only for bounded correctness smokes, not exponential K-best/EP performance sweeps. The shipped GPU profile notes K-best OOM for batch 20 (`configs/bler_4ue_cdl_gpu.toml:3-8`).
- K-best requires physical RX antennas at least equal to aggregate streams in both waveform implementations (`receiver.py:283-285,888-890`); DFT-s-OFDM K-best additionally requires one BS endpoint (`receiver.py:283-285`).
- Delay estimation reports a diagnostic bulk delay; it is not capture time alignment. BLER's optional zero-error stop skips all later configured SNR entries after the first zero-error point, so ordered increasing SNR is an assumption (`bler.py:137-141,274-287`).
- BLER point `runtime_s` covers point work, while model setup and export occur outside it; process wall time must be recorded separately (`bler.py:149-188,197-268,326-375`). Device memory sampled by system telemetry is total GPU use, not PyTorch allocator peak.
- Checked-in unit/integration tests are correctness/control evidence only. Low-frame smoke BLER and reduced paired validation are diagnostic, not statistical release acceptance or performance claims.

## Test map

- Config/topology/noise/device/import: `tests/unit/test_tx_topology_config.py`, `test_channel_config.py`, `test_noise.py`, `test_device.py`, `test_matlab_h5.py`, `test_simulation_config.py`.
- Simulation/detector/prior: `tests/unit/test_bler_sweep.py`, `test_dmrs_lmmse.py`, `test_soft_mmse_pic.py`, `test_estimator_validation.py`.
- Sionna end-to-end: `tests/integration/test_pusch_transmitter.py`, `test_mimo_transmitter.py`, `test_cdl_channel.py`, `test_mimo_channel.py`, `test_pusch_receiver.py`, `test_mimo_receiver.py`, `test_dmrs_lmmse.py`, `test_mimo_bler.py`.
- Web API/RX import and matrix: `tests/integration/test_web_rx_import.py`, `test_web_estimator_matrix.py`.

The full standard-library unittest discovery command is the release correctness gate; these file locations alone do not assert that it passed.
