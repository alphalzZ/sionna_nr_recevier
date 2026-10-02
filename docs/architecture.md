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

The legacy scalar DFT-s-OFDM DMRS estimator is restricted to the four-user,
single-layer, one-port non-codebook capture profile; multi-port codebook
profiles use the generic layer-domain MIMO DMRS estimator.
The regression profile remains four single-layer UEs on one physical TX port
each and four RX antennas. MATLAB fixture files and their specialized readers
are not generalized or regenerated.

