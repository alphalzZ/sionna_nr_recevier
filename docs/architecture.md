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

1. `rt_config.py` validates four single-layer UE positions/order, array, noise, optional parameterized geometry and Web resource ceilings. `nr-pusch-rt-channel` traces or reuses a verified content-addressed static snapshot under `outputs/rt_snapshots/`; its key binds RT/TX resource and DMRS geometry, scene asset hashes, runtime versions/variant and snapshot implementation. The NPZ/JSON keeps path data in element space and `h_ant`/`h_beam` axes `[element,user,fft_bin]` and `[beam,user,fft_bin]`.
2. `beamforming.py` applies one shared `Wᴴ` across each UE column to form the four receive beams. `noise.py` adds shared array noise plus independent post-combiner noise with covariance `R_eta = sigma_a² WᴴW + sigma_v² I`; neither propagation nor signal paths are resampled per detector.
3. `NrPuschRtBeamChannel` maps the immutable snapshot into the existing frequency-grid or time-IQ receiver contract. `nr-pusch-beam-bler` applies UE power once, reuses the same per-frame payload/noise across receiver arms, diagonally scales independent-beam observations or whitens joint observations, then invokes the existing PUSCH decoder.
4. The BLER CSV/JSON records conditional fixed-snapshot results, per-UE and aggregate rows, covariance and coordinate basis. `nr-pusch-rx` can replay the matching scaled capture; mixing physical, diagonal-scaled and whitened CSI is invalid.
5. `rt_scene_assets.py` resolves the three project scenes and all15 pinned Sionna-RT package scenes, generates parameterized ground/wall meshes, or validates the unchanged restricted XML＋ASCII-PLY import format. Each job owns only the referenced scene assets, bound by canonical bundle hash; parameterized geometry remains limited to project ground/wall.
6. The RT Web FIFO worker reuses only a hash/version-matching channel snapshot; on cache miss, the internal prepare child traces one, then applies the same Web acceptance gate and doubled-sample mesh convergence check on both hit and miss. BLER always reruns against the validated fixed snapshot.
7. Successful RT Web jobs are atomically archived with configs, scene copy, snapshot, validation, CSV/JSON, progress/log/job metadata under `outputs/rt_runs/<job-id>/`. Failed/cancelled jobs are not archived; CDL job storage remains unchanged.
8. Web acceptance requires FD/TD relative RMS≤0.001, CP/window and direct-CFR truncation≤1%, plus doubled-sample convergence for meshes. The existing strict `1e-5` result remains separate and visible; every Web BLER is labeled conditional on one static geometry/snapshot, not strict time-domain equivalence or a statistical city benchmark.
9. The RT Web scene-preview endpoint reuses the validated built-in, parameterized or imported scene assets; `rt_scene_preview.py` adds display-only BS/UE markers at the configured XYZ coordinates and renders 640×400, 16-sample oblique/top PNGs. The endpoint performs no propagation-path tracing; the browser keeps the XY view and lists full coordinates alongside the color legend.

RT uses Sionna/Sionna RT 2.2.0 and package-owned or explicitly validated scene assets; it never falls back to CDL or the CDL tap-power prior. CDL profiles remain available as the default Web backend and keep their existing workflow.

The legacy scalar DFT-s-OFDM DMRS estimator is restricted to the four-user,
single-layer, one-port non-codebook capture profile; multi-port codebook
profiles use the generic layer-domain MIMO DMRS estimator.
The regression profile remains four single-layer UEs on one physical TX port
each and four RX antennas. MATLAB fixture files and their specialized readers
are not generalized or regenerated.

## Web history and BLER chart

`DELETE /api/runs/{id}` removes one terminal run; `DELETE /api/runs` accepts `{"ids":[...]}` for a selected batch. Queued/running jobs are rejected before any batch deletion. Removing a job deletes its `runs_dir/<id>` and per-job `outputs/rt_runs/<id>` archive, but leaves shared `outputs/rt_snapshots/` intact.

History controls disable deletion for active rows and require confirmation; deleting the selected current job clears its result view. BLER legend checkboxes independently hide/show estimator/detector curves, “显示全部” restores all curves, and the measurement table is not filtered.

