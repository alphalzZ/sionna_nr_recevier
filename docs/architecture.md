# Repository structure and test boundaries

TX and RX exchange IQ or OFDM grids through NPZ/JSON artifacts; MATLAB H5
ingestion remains a separate fixed four-user capture adapter.

## Signal path

1. `config.py` parses shared PUSCH parameters plus each UE's `dmrs_ports`.
   Sionna verifies its native one-to-four-layer PUSCH constraints.
2. `transmitter.py` returns separate physical antenna waveforms and one TB
   payload per UE; the optional codebook precoder maps layers to antennas.
3. `channel.py` draws independent CDL links per UE and combines every physical
   TX antenna at each configurable BS RX antenna. Frequency CSI uses
   `[batch,1,rx_antenna,user,tx_antenna,symbol,fft_bin]`; time taps use
   `[batch,user,rx_antenna,tx_antenna,time,tap]`.
4. `noise.py` adds variance calibrated separately for every batch/RX antenna.
5. `receiver.py` projects perfect physical CSI through Sionna's codebook when
   needed, or jointly estimates effective layer CSI from DMRS. Detectors
   return `[batch,user,layer,coded_bit]`; Sionna interleaves layers back into
   one TB codeword per UE before decoding.
6. `bler.py` counts a block error whenever UE CRC fails or decoded payload
   differs. `estimator_validation.py` compares effective layer CSI on data RE.

## Separate Sionna RT four-beam experiment path

The static RT experiment is an isolated path, not a replacement for the CDL regression above:

1. `rt_config.py` validates four single-layer UE positions/order, array, noise, optional parameterized geometry and Web resource ceilings; `nr-pusch-rt-channel` traces the selected package, generated or imported scene and writes a version-2 NPZ/JSON snapshot with scene-bundle identity. `path_a`, `path_tau_s`, and `path_valid` stay in element space; `h_ant` and `h_beam` retain `[element,user,fft_bin]` and `[beam,user,fft_bin]` axes.
2. `beamforming.py` applies one shared `Wᴴ` across each UE column to form the four receive beams. `noise.py` adds shared array noise plus independent post-combiner noise with covariance `R_eta = sigma_a² WᴴW + sigma_v² I`; neither propagation nor signal paths are resampled per detector.
3. `NrPuschRtBeamChannel` maps the immutable snapshot into the existing frequency-grid or time-IQ receiver contract. `nr-pusch-beam-bler` applies UE power once, reuses the same per-frame payload/noise across receiver arms, diagonally scales independent-beam observations or whitens joint observations, then invokes the existing PUSCH decoder.
4. The BLER CSV/JSON records conditional fixed-snapshot results, per-UE and aggregate rows, covariance and coordinate basis. `nr-pusch-rx` can replay the matching scaled capture; mixing physical, diagonal-scaled and whitened CSI is invalid.
5. `rt_scene_assets.py` resolves packaged geometry, generates parameterized ground/wall meshes, or validates a restricted XML＋ASCII-PLY bundle. Snapshots and Web jobs bind the referenced files by canonical bundle hash; each job owns its copied scene tree for later replay.
6. The Web dashboard adds an RT backend to the existing FIFO worker: an internal prepare child traces and checks `web-frequency-v1`, then the BLER child reuses the exact validated snapshot. Presets cover LoS, ground, ground+wall and CP-LoS; advanced editing supports fixed parameters and the restricted scene package, not arbitrary Mitsuba projects or 3D drag/drop.
7. Web acceptance requires FD/TD relative RMS≤0.001, CP/window and direct-CFR truncation≤1%, plus doubled-sample convergence for meshes. The existing strict `1e-5` result remains separate and visible; every Web BLER is labeled conditional on one static geometry/snapshot, not strict time-domain equivalence or a statistical city benchmark.

RT uses Sionna/Sionna RT 2.2.0 and package-owned or explicitly validated scene assets; it never falls back to CDL or the CDL tap-power prior. CDL profiles remain available as the default Web backend and keep their existing workflow.

The legacy scalar DFT-s-OFDM DMRS estimator is restricted to the four-user,
single-layer, one-port non-codebook capture profile; multi-port codebook
profiles use the generic layer-domain MIMO DMRS estimator.
The regression profile remains four single-layer UEs on one physical TX port
each and four RX antennas. MATLAB fixture files and their specialized readers
are not generalized or regenerated.

