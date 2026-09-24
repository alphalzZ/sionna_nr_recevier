# Repository structure and test boundaries

This repository keeps TX, RX, and IQ ingestion in one small Python package. TX and RX exchange IQ files plus JSON manifests; no receiver code depends on a live transmitter object.

## Current modules

- `config.py`: TOML parsing, validation, and conversion to one Sionna PUSCH config per UE.
- `transmitter.py`: 4-UE/one-layer Sionna waveform adapter.
- `artifacts.py`: portable IQ/payload archive and resolved-run manifest.
- `cli/tx.py`: command-line sender.

Later receiver work will add independent `iq/` and `receiver/` modules. MATLAB H5 schema and its adapter are intentionally deferred until the actual fixture is available.

## Sionna-level test nodes

1. Config and UE ordering into the Sionna wrapper.
2. Sionna frequency-grid output when a fixture contains that stage.
3. DFT-s-OFDM adapter output when transform precoding is implemented.
4. Final time-domain IQ versus MATLAB reference, after applying fixture-defined alignment/scaling rules.
5. Sionna receiver input IQ to per-UE payload bits and CRC status.
6. Local TX-to-RX loopback through a deterministic channel.

The tests target public composite interfaces and repository adapters. Sionna's internal CRC, LDPC, DMRS, channel-estimation, and detector substeps are not duplicated as separate repository unit tests.

