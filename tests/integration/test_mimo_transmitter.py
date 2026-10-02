from dataclasses import replace
from pathlib import Path
import unittest

import torch

from nr_pusch.config import TxSettings, UserSettings
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class MimoTransmitterTest(unittest.TestCase):
    def test_four_and_eight_stream_grids_keep_physical_antennas_and_tb_bits(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        for users, waveform in ((1, "cp_ofdm"), (2, "cp_ofdm"), (2, "dft_s_ofdm")):
            with self.subTest(users=users, waveform=waveform):
                settings = replace(
                    base,
                    pusch=replace(base.pusch, waveform=waveform, num_layers=4,
                                  num_antenna_ports=4, dmrs_length=2 if users == 2 else 1,
                                  n_size_bwp=6, mcs_index=4),
                    users=tuple(UserSettings(f"ue{i}", i + 1,
                                             tuple(range(4 * i, 4 * (i + 1))))
                                for i in range(users)),
                )
                tx = NrPuschTx(settings, device="cpu")
                result = tx.generate(batch_size=1, seed=7)
                self.assertEqual(result.iq.shape[:3], (1, users, 4))
                self.assertEqual(result.frequency_grid.shape[:3], (1, users, 4))
                self.assertEqual(result.bits.shape, (1, users, tx.transport_block_size))
                self.assertEqual(result.metadata["total_streams"], users * 4)
                mask = tx._tx_freq.pilot_pattern.mask
                self.assertEqual(mask.shape[:2], (users, 4))
                self.assertEqual(sum(len(u.dmrs_ports) for u in settings.users), users * 4)
                dmrs_symbols = tx.configs[0].dmrs_symbol_indices
                for user in range(users):
                    for antenna in range(4):
                        self.assertTrue(torch.any(result.frequency_grid[0, user, antenna,
                                                                         dmrs_symbols].abs() > 0))

    def test_three_layer_codebook_emits_four_physical_antennas(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        settings = replace(base,
                           pusch=replace(base.pusch, waveform="cp_ofdm", num_layers=3,
                                         num_antenna_ports=4, precoding="codebook",
                                         n_size_bwp=6, mcs_index=4),
                           users=(UserSettings("ue0", 1, (0, 1, 2)),))
        tx = NrPuschTx(settings, device="cpu")
        generated = tx.generate(seed=8)
        self.assertEqual(generated.frequency_grid.shape[:3], (1, 1, 4))
        self.assertEqual(generated.bits.shape[-1], tx.transport_block_size)


if __name__ == "__main__":
    unittest.main()
