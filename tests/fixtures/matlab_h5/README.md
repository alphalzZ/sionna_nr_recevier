# MATLAB H5 reference fixtures

`TxTestVector.h5` is the supplied MATLAB transmitter reference. Each UE has `ueNTxBits`, `ueNFreqIQ_real`, and `ueNFreqIQ_imag` datasets. h5py presents the frequency arrays as `[OFDM symbol, layer, subcarrier]`; split real and imaginary datasets are combined as complex values. See `nr_pusch.iq.read_matlab_tx_reference`.

The H5 descriptions say the frequency-domain modulation symbols have already undergone DFT processing. The current TOML still selects `cp_ofdm`, so this fixture cannot be treated as a final CP-OFDM resource-grid gold vector without clarifying the waveform stage. Keep the supplied file unchanged.
