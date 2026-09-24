# Transmitter interface and initial coverage

The initial sender wraps Sionna 2.0.1 `PUSCHTransmitter` with four `PUSCHConfig` objects. Each config represents one UE with one layer and one transmit antenna. The Sionna output keeps the UE signals separate; RF/channel superposition belongs to a later channel harness.

The transmitter currently exports Sionna's time-domain CP-OFDM output, Sionna's pre-modulation resource grid, and the payload bits used to generate them. NPZ axis order is `[batch, user, tx_antenna, sample]` for `iq`, `[batch, user, tx_antenna, symbol, subcarrier]` for `frequency_grid`, and `[batch, user, transport_block_bit]` for `bits`. A JSON sidecar records the resolved TOML settings and library versions.

For DFT-s-OFDM, the adapter applies a unitary DFT on each data-bearing OFDM symbol inside the configured contiguous BWP, replaces Sionna's CP-OFDM pilots with type-1 low-PAPR DMRS, and runs Sionna's OFDM modulator on the resulting grid. The initial DMRS profile supports no group/sequence hopping, DMRS ports 0--3, and one DMRS symbol. The Zadoff-Chu sequence, port comb, and frequency cover were matched to the supplied MATLAB vector. Since the pinned Sionna composite does not accept transform-precoding MCS tables, the adapter maps to a native MCS with the same modulation order and code rate; the selected project MCS remains in the manifest.
