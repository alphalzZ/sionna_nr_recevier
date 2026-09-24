from pathlib import Path
import unittest

import numpy as np
import torch

from nr_pusch.config import TxSettings
from nr_pusch.iq import read_matlab_tx_reference
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class PuschTransmitterTest(unittest.TestCase):
    def test_configured_four_user_transmitter_generates_separate_user_waveforms(self):
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        tx = NrPuschTx(settings, device="cpu")
        result = tx.generate(batch_size=1, seed=13)

        self.assertEqual(result.iq.ndim, 4)
        self.assertEqual(tuple(result.iq.shape[:3]), (1, 4, 1))
        self.assertEqual(tuple(result.frequency_grid.shape[:4]), (1, 4, 1, 14))
        self.assertEqual(tuple(result.bits.shape), (1, 4, 28168))
        self.assertEqual(result.sample_rate_hz, 18_000_000)
        self.assertEqual(settings.pusch.waveform, "dft_s_ofdm")
        self.assertEqual(settings.effective_sionna_mcs(), (1, 21))
        self.assertAlmostEqual(float(tx.configs[0].tb.target_coderate.item()), 0.6015625)

    def test_matlab_payload_and_frequency_grid_match_transmitter(self):
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        tx = NrPuschTx(settings, device="cpu")
        reference = read_matlab_tx_reference(ROOT / "tests" / "fixtures" / "matlab_h5" / "TxTestVector.h5")
        ref_bits = torch.from_numpy(reference.bits[None, ...].astype("float32"))
        result = tx.generate(batch_size=1, bits=ref_bits)

        torch.testing.assert_close(result.bits, ref_bits)
        generated_grid = result.frequency_grid[0].permute(0, 2, 1, 3).detach().cpu().numpy()
        np.testing.assert_allclose(generated_grid, reference.frequency_grid, rtol=1e-6, atol=2e-6)

    def test_transform_precoded_dmrs_resource_pattern_matches_reference(self):
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        tx = NrPuschTx(settings, device="cpu")
        generated = tx.generate(batch_size=1, seed=19)
        reference = read_matlab_tx_reference(ROOT / "tests" / "fixtures" / "matlab_h5" / "TxTestVector.h5")

        grid = generated.frequency_grid.detach().cpu().numpy()[0]
        generated_zeros = np.count_nonzero(np.abs(grid) < 1e-12, axis=(1, 3))
        reference_zeros = np.count_nonzero(np.abs(reference.frequency_grid) < 1e-12, axis=(2, 3))
        np.testing.assert_array_equal(generated_zeros, reference_zeros)


if __name__ == "__main__":
    unittest.main()
