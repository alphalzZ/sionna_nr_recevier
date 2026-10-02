from dataclasses import replace
from pathlib import Path
import unittest

import torch
from sionna.phy.nr import PUSCHTransmitter
from sionna.phy.ofdm import OFDMModulator

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
                    pusch=replace(
                        base.pusch, waveform=waveform, num_layers=4,
                        num_antenna_ports=4, dmrs_length=2 if users == 2 else 1,
                        n_size_bwp=6, mcs_index=4,
                        dmrs_beta=2**0.5 if waveform == "cp_ofdm" else base.pusch.dmrs_beta,
                    ),
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
                                         n_size_bwp=6, mcs_index=4, dmrs_beta=2**0.5),
                           users=(UserSettings("ue0", 1, (0, 1, 2)),))
        tx = NrPuschTx(settings, device="cpu")
        generated = tx.generate(seed=8)
        self.assertEqual(generated.frequency_grid.shape[:3], (1, 1, 4))
        self.assertEqual(generated.bits.shape[-1], tx.transport_block_size)

    def test_cp_ofdm_frequency_grid_and_time_iq_match_native_sionna(self):
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_cp_2ue_2layer.toml")
        tx = NrPuschTx(settings, device="cpu")
        bits = torch.arange(2 * tx.transport_block_size, dtype=torch.int32)
        bits = (bits % 2).reshape(1, 2, tx.transport_block_size).float()

        sent = tx.generate(bits=bits)
        native_freq = PUSCHTransmitter(
            settings.to_sionna_configs(), return_bits=False, output_domain="freq",
            device="cpu",
        )
        native_time = PUSCHTransmitter(
            settings.to_sionna_configs(), return_bits=False, output_domain="time",
            device="cpu",
        )
        torch.testing.assert_close(
            sent.frequency_grid, native_freq(bits), rtol=0, atol=1e-5
        )
        torch.testing.assert_close(sent.iq, native_time(bits), rtol=0, atol=1e-5)
        modulated = OFDMModulator(
            tx._tx_freq.resource_grid.cyclic_prefix_length, device="cpu"
        )(sent.frequency_grid)
        torch.testing.assert_close(sent.iq, modulated, rtol=0, atol=1e-5)

    def test_cp_type2_dmrs_symbol_contains_data_and_pilot_re(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_cp_2ue_2layer.toml")
        settings = replace(
            base,
            pusch=replace(
                base.pusch,
                dmrs_config_type=2,
                dmrs_num_cdm_groups_without_data=1,
                dmrs_beta=1.0,
                num_layers=1,
                num_antenna_ports=1,
            ),
            users=(UserSettings("ue0", 1, (0,)),),
        )
        tx = NrPuschTx(settings, device="cpu")
        result = tx.generate(seed=11)
        mask = tx._tx_freq.pilot_pattern.mask[0, 0]
        dmrs_symbol = tx.configs[0].dmrs_symbol_indices[0]
        self.assertTrue(torch.any(mask[dmrs_symbol]))
        self.assertTrue(torch.any(~mask[dmrs_symbol]))
        self.assertEqual(result.metadata["frequency_grid_stage"], "Sionna CP-OFDM resource grid")


if __name__ == "__main__":
    unittest.main()
