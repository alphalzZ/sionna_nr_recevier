# Release 0.1 simulation and GPU baseline runbook

## Purpose and evidence levels

This runbook freezes the Release 0.1 correctness coverage and a conservative, comparable GPU throughput workload. Run it from the repository root using `/home/le-lei/workspace/test/.venv`. Do not edit active profiles in `configs/` or supplied MATLAB/H5 fixtures. All generated evidence stays beneath `outputs/release-0.1-baseline/` (NPZ is globally ignored; keep text, manifests, configs and hashes reviewable).

Three kinds of evidence remain distinct:

1. **Correctness gate:** complete standard-library unittest suite plus GPU functional smokes for shipped behavior.
2. **Diagnostic estimator validation:** reduced paired DFT/CP jobs train separate priors and preserve frame-level results. They do not satisfy full statistical acceptance.
3. **Measured performance:** only the two fixed frequency-domain CDL-A workloads and three fixed seeds below. BLER at 8 or 16 frames is descriptive; the run is not maximum-throughput tuning.

Current package metadata is `nr-pusch-lab` 0.1.0, Python `>=3.11`, Sionna 2.2.0 and Sionna RT 2.2.0 (`pyproject.toml`). Pre-migration inventory: Python 3.14.4, Sionna/Sionna RT 2.0.1, PyTorch 2.13.0+cu130, NumPy 2.5.2, h5py 3.16.0, Mitsuba 3.8.0, Dr.Jit 1.3.1, RTX 3060 Laptop GPU (6,064,832,512-byte VRAM), driver CUDA 13.2, Ryzen 5 5600H (6C/12T), 16 GiB RAM. The 2.2.0 migration resolved Mitsuba 3.9.1 and Dr.Jit 1.5.0 without replacing PyTorch; recapture exact environment before freezing. These changes do not rewrite historical CDL measurements. The prior GPU snapshot showed 26% utilization and 870 MiB allocated, so it did not satisfy the idle requirement. Never terminate unrelated GPU processes.

`bler.runtime_s` measures each SNR point's simulation work including TX/channel/noise/RX/counters, but excludes model setup and export (`src/nr_pusch/bler.py:149-189,195-270,322-371`). Report frames/s and TB/s from this point runtime and `/usr/bin/time -v` wall time separately. One-second `nvidia-smi` memory is total device usage, not PyTorch allocator peak.

## Run order and immutable evidence

### 1. Initialize identity and folders

```bash
cd /home/le-lei/workspace/test/sionna_nr_recevier
export PYTHONPATH=src
BASE=outputs/release-0.1-baseline
mkdir -p "$BASE"/{configs,telemetry,logs,functional/{tx,channel,rx,web-runs},performance/{dft_4ue,cp_2ue_2layer},estimator-validation/{dft_4ue,cp_2ue_2layer}}
git rev-parse HEAD > "$BASE/source_revision.txt"
git diff --binary HEAD -- src pyproject.toml > "$BASE/source.patch"
```

If `source.patch` is empty, record that fact and remove only the empty generated patch. Capture `python --version`, relevant package versions (`pip freeze` or exact package versions), `nvidia-smi`, and SHA-256 values for every input profile in the manifest. Copy all TX/CDL/RX/BLER/validation source TOMLs used into `configs/`; do not point measured runs at mutable profiles. Record the exact git revision plus optional patch digest and config digests as the baseline identity.

When a measured `dmrs-lmmse` arm is planned, snapshot the matching, already-published priors from the local `configs/tap_power_prior/` registry under `$BASE/configs/tap_power_prior/`; include their compatibility metadata and SHA-256 values in the manifest. Do not substitute diagnostic candidates. If a compatible accepted prior is unavailable, mark the affected `dmrs-lmmse` rows blocked rather than claiming a complete sweep.

### 2. GPU idle preflight and telemetry

Require `nvidia-smi` utilization **≤5% continuously for 60 seconds** before any measured run. Record model, driver, utilization, memory used/total, temperature and power at start; do not kill other processes. If the precondition cannot be met, run correctness work as possible, preserve diagnostic outputs, mark measured performance **contaminated/not frozen**, and do not present it as the baseline.

After passing preflight, start one-second telemetry and retain it through performance work:

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -l 1 --filename="$BASE/telemetry/gpu.csv" &
GPU_TELEMETRY_PID=$!
printf '%s\n' "$GPU_TELEMETRY_PID" > "$BASE/telemetry/gpu.pid"
```

Stop only this recorded telemetry PID after all workloads finish; preserve the complete CSV and check that its final sample covers the last run. Use `/usr/bin/time -v -o <run>.time` for every BLER process. Run long simulations with no host/tool execution deadline (`timeout=0`); do not impose a guessed wall-clock cutoff or terminate a running profile. A failed or interrupted profile is incomplete and cannot be frozen.

### 3. Correctness gate

Run the entire suite once; this is the only full-suite invocation for this plan:

```bash
PYTHONPATH=src /home/le-lei/workspace/test/.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v > outputs/release-0.1-baseline/logs/unittest.log 2>&1
```

Capture exit status, exact test count and complete log. Failure blocks release acceptance; do not replace the complete suite with selected tests.

### 4. GPU functional matrix

Run every shipped `configs/pusch_*.toml` profile through `nr-pusch-tx` on `cuda:0`, `--batch-size 1 --seed 13`; save NPZ and JSON sidecar under `functional/tx/`. Check each archive's shapes, dtypes, axes, sample rate and TOML settings against its sidecar. This glob includes all eight shipped TX profiles and no RX-only TOML.

Exercise channel coverage with one-frame deterministic TX input on GPU:

- Use CDL-A through CDL-E for both `--domain time` and `--domain frequency` where that input representation applies: IQ → time; frequency grid → frequency. For CDL-B–E, copy the checked-in CDL-A profile and change only `model`.
- Include 1×UT/4×BS, 2×UT/4×BS and 4×UT/8×BS geometries using matching TX and CDL profiles. Include one fixed nonzero-speed case with `min_speed_mps=max_speed_mps>0`.
- Verify output keys, JSON sidecar settings, domain, axes, array shapes/dtypes, per-UE output and frequency CSI/time taps as applicable. Retain every output and command log under `functional/channel/`.
- Exercise deterministic AWGN using the fixed-seed time-IQ and resource-grid entrypoints; use existing unit/integration tests for exact statistics/shape behavior. Retain a GPU functional artifact or log for both paths.

Exercise RX and detector coverage on GPU:

- Run the DFT-S-OFDM baseline and CP-OFDM profiles across estimator arms `[dmrs, dmrs-lmmse, perfect]` and detector arms `[lmmse, lmmse-sic, k-best, ep, mmse-pic, soft-mmse-pic]`, with compatible settings/priors. Run one-frame 8-stream LMMSE and MMSE-PIC correctness smokes; do not include 8-stream K-best/EP in performance sweeps.
- Cover type-2 DMRS, DMRS length 1 and 2, nonzero additional positions and codebook mapping using compatible shipped settings or temporary snapshots; do not change checked-in profiles. Exercise perfect CSI and DMRS-LMMSE through RX validation paths. Record every unsupported combination or blocked row with exact error/log; no silent skip.
- Decode the repository's actual supplied MATLAB H5 RX fixture using `nr-pusch-rx --rx-config configs/rx_pusch_4ue.toml --input tests/fixtures/matlab_h5/RxTestVector.h5 --input-format matlab-h5 --output "$BASE/functional/rx/matlab-reference.npz" --device cuda:0`. Preserve the JSON reference comparison, CRC and bit results. Record absent optional datasets as absent; never synthesize them.

### 5. Live web surface

Use the existing root `configs/` profiles directly; `configs/bler_release_smoke.toml` supplies the one-SNR GPU browser smoke parameters. Start:

```bash
nr-pusch-web --host 127.0.0.1 --port 8765 \
  --config-dir configs --runs-dir "$BASE/functional/web-runs"
```

Use a real Chromium browser session on the dashboard (not API-only evidence): load dashboard; list/read/save a temporary copied profile; submit a one-frame GPU BLER run; poll to completion; download and validate its CSV and JSON; decode the H5 fixture through the RX endpoint with `device="cuda:0"`; download and inspect decoded NPZ and JSON. Capture browser-visible proof, server logs and download artifacts under `functional/web/`. Stop only the server launched for this test after capturing logs.

Run-history acceptance also deletes disposable terminal jobs singly and in batches, verifies queued/running jobs reject deletion without partial batch removal, confirms per-job output archives disappear while shared RT snapshots remain, and checks a deleted current job no longer remains displayed.

For chart filtering, open a completed multi-series run, uncheck all but one estimator/detector line, confirm only that curve remains while the measured-row count is unchanged, then use “显示全部” to restore every line.

RT 的 Web 验收是另一条功能/正确性 smoke，不替代以上 CDL/RX 浏览器回归：在隔离的 loopback 服务和临时 runs 目录中，实测 LoS、ground+wall 与 CP-LoS quick presets、参数化/受限 ZIP 场景、取消、history 和结果下载；保存 Web validation report、场景包和固定快照 replay。浏览器验收使用临时 quick profile、每点最多2帧，仅作诊断，不运行统计 profile。Web job 仅在 `web-frequency-v1` 通过后启动 BLER；ground+wall 可在 Web RMS≤0.001 下运行，但必须持续显示 strict FD/TD `1e-5` 未通过的 warning。另须通过真实浏览器验证至少一个 Sionna 随包场景在斜视/顶视下的 640×400、16-sample 几何预览，检查 BS/UE 图例、完整 XYZ 坐标、原 XY 参考和无传播路径计算；非 loopback 部署仍需外部认证和访问控制。

The checked-in `blender_scene/test_scene/sionna_rt_export.zip` is also a required imported-scene smoke: retain the existing upload limits, import its two ASCII PLY shapes, verify the 3,348/4,768 building mesh and 4/2 ground mesh, and inspect the 640×400 oblique/top previews with all four UE markers and the BS position. Do not run propagation or BLER as part of the visual-only preview.

### 6. Reduced paired estimator validation and exact priors

Create workload-specific immutable validation snapshots for DFT and CP. Run separately:

```bash
nr-pusch-estimator-validation --tx-config <TX-snapshot> \
  --channel-config <CDL-snapshot> --validation-config <validation-snapshot> \
  --output "$BASE/estimator-validation/<profile>/validation.json" \
  --device cuda:0 --prior-realizations 32 --frames-per-snr <cap> \
  --prior-dir "$BASE/estimator-validation/<profile>/prior-store"
```

Use `--frames-per-snr 16` unless 8 is selected before any measured performance run; the same choice applies to both profiles. Keep the configured 64-frame high-SNR check. Preserve summary JSON, per-frame NPZ and the diagnostic prior NPZ in each profile directory. The CLI publishes a marked prior into `--prior-dir` only when `acceptance_gate.passed` is true, so keep these reduced runs isolated with `--prior-dir "$BASE/estimator-validation/<profile>/prior-store"`; never use that store for the measured sweep. Reduced samples remain diagnostic even if the configured gate passes and a marked file is emitted; they do not establish the planned full 3000-frame acceptance evidence. The performance sweep instead uses the previously accepted prior snapshots captured in step 1, explicitly passed from `$BASE/configs/tap_power_prior/`. Changing only MCS table/index does not invalidate the channel tap-power prior, while any other compatibility-field mismatch remains an error. The checked-in full validation profile historically took 74m56 on an RTX 3060.

### 7. Frozen workload choice

Select one frame cap **before starting any measured run** and record it in every immutable config and manifest:

- Default: `max_frames_per_snr=16`; every point must run exactly 16 frames.
- If measured runtime requires a smaller workload, select `8` before any measured run and use it uniformly for every profile, detector, estimator, SNR and seed.
- Never begin at 16 then lower cap later: an 8-frame run is not comparable to an already frozen 16-frame baseline.

Use identical profile/estimator/detector settings across repetitions; only the seed changes. Disable early stopping, use batch 1, and verify `frames == cap` for every point. At 16 frames, each profile has 3 estimators × 6 detectors × 3 SNRs = 54 points/seed, 864 frames/seed, 2,592 frames/profile across three seeds, 5,184 total frames. At 8 frames, total across both profiles is 2,592 frames.

### 8. Performance profiles and run commands

Workloads:

| Name | TX / CDL | Topology and fixed profile |
|---|---|---|
| `dft_4ue` | `pusch_4ue.toml` + `cdl_38_901_4x4.toml` | DFT-s-OFDM, four single-layer UEs, MCS 20, 50-RB BWP, CDL-A frequency domain. |
| `cp_2ue_2layer` | `pusch_cp_2ue_2layer.toml` + `cdl_38_901_2tx_4rx.toml` | CP-OFDM, two UEs × two layers, MCS 20, 12-RB BWP, CDL-A frequency domain. |

For each workload, create one immutable BLER TOML per seed `[20260924, 20260925, 20260926]`, with:

```toml
snr_db = [20.0, 25.0, 30.0]
batch_size = 1
max_frames_per_snr = <selected 16 or 8 cap>
target_block_errors = 1000000000
seed = <one listed seed>
num_decoder_iterations = 20
channel_estimator = "dmrs"
channel_estimators = ["dmrs", "dmrs-lmmse", "perfect"]
detectors = ["lmmse", "lmmse-sic", "k-best", "ep", "mmse-pic", "soft-mmse-pic"]
detector_parameters = { "k-best" = 16, "ep" = 10, "mmse-pic" = 4, "soft-mmse-pic" = 1 }
detector_damping = 0.25
device = "cuda:0"
channel_domain = "frequency"
l_min = -6
max_delay_spread_s = 3e-6
stop_at_zero_bler = false
# 先验由共享 registry 按当前 TX/CDL 兼容性解析；不要在 BLER TOML 中设置先验路径
```

Do not add retries, omit detectors, or change settings silently. The low-count matrix is descriptive, not statistically significant. Do not interpret shared seeds as statistically paired BLER observations across detector arms.

Run all six profile/seed combinations; retain resolved TOML snapshots, raw CSV/JSON, JSONL progress, logs and wall-time records:

```bash
/usr/bin/time -v -o "$BASE/performance/<profile>/seed-<seed>/run.time" \
  nr-pusch-bler --tx-config "$BASE/configs/<tx>.toml" \
  --channel-config "$BASE/configs/<cdl>.toml" \
  --simulation-config "$BASE/performance/<profile>/seed-<seed>/simulation.toml" \
  --output "$BASE/performance/<profile>/seed-<seed>/results.csv" \
  --progress-jsonl "$BASE/performance/<profile>/seed-<seed>/progress.jsonl" \
  --device cuda:0 --prior-dir "$BASE/configs/tap_power_prior"
```

The DFT and CP profiles are the only throughput baseline. CDL-B–E, moving speed, time-domain channel, alternate topology, DMRS/codebook variants and 8-stream cases are correctness/coverage only; they have no Release 0.1 throughput claim.

### 9. Result validation and summaries

For every JSON point verify: requested estimator × detector × SNR tuple; `frames` equals the selected cap; `transport_blocks` equals frames × user count; `device` identifies `cuda:0`; run completed without OOM; seed and resolved profile are preserved. Any mismatch, incomplete seed, contaminated preflight or missing telemetry means performance is not frozen.

Compute per-point:

- `frames_per_second = frames / runtime_s`
- `transport_blocks_per_second = transport_blocks / runtime_s`

Retain frames, TBs, CRC failures, TB block errors, bit errors, bit count, BLER, CRC-failure rate, BER, point runtime, per-process wall time, and device utilization/memory observations. Build `performance_summary.csv` with every individual seed row and mean/median throughput summaries. Point runtime excludes setup/export; wall time includes process overhead. Sampled GPU memory is device-wide usage, not PyTorch peak allocation.

### 10. Final manifest, report, coverage and hashes

`feature_coverage.csv` must describe each planned scenario from actual test/smoke results: feature/scenario, exact test or smoke command, pass/blocked status, artifact/log path and limitation. Source support alone is not a pass. Include every blocked row and evidence; do not omit planned work.

Write `manifest.json` and `README.md` with exact source revision and optional patch hash, all input-config SHA-256s, Python/package/GPU/driver versions, all commands and resolved parameters, chosen frame cap, seed list, per-run exit/completion status, wall times, telemetry summary, artifact paths and known coverage/statistical limitations. State explicitly that low-count BLER and shortened estimator validation are diagnostic only and that no release-adoption or statistical claim follows.

Finish `checksums.sha256` only after output artifacts have settled. Hash source/config snapshots, environment inventory, logs, reports, CSV/JSON, telemetry and all other retained files; include patch digest if nonempty. Preserve generated NPZ locally under the output tree even though globally ignored. Never modify or delete the user-owned `.omp/` tree.

## Separate RT experimental lane

Sionna RT 2.2.0 four-beam remains a separate correctness lane, not a Release 0.1 CDL throughput workload. Preserve historical strict CLI gate reports and their exact pass/fail meaning. CLI research statistics require their documented strict `all` acceptance; the Web worker has a separate fixed-snapshot `web-frequency-v1` admission gate (FD/TD RMS≤0.001, CP/window/CFR bounds≤1%, mesh convergence) and retains strict `1e-5` as a visible independent quality field. Passing the Web gate authorizes only conditional BLER for that static snapshot; it does not rewrite a strict failure, establish dynamic-city equivalence, or create a statistical benchmark. Do not claim the 500-frame Web budget was measured unless that run actually completed.

## Acceptance gates

- Full unittest suite passes, with exact test count and log captured.
- Every planned functional row is pass or explicitly blocked with command/output evidence; no unreported skips.
- Every performance point uses the preselected fixed cap, correct seed/profile and successful GPU device marker; no OOM or missing run.
- Idle GPU requirement and one-second telemetry are retained; otherwise performance status is contaminated/not frozen.
- Exact config/source identity and final checksums are available.
- Report distinguishes correctness, diagnostic statistical evidence and measured performance, and does not claim significance for 8/16-frame BLER.
