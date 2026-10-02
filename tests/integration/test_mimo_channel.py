from dataclasses import replace
from pathlib import Path
import unittest

import torch
import sionna.phy

from nr_pusch.channel import NrPuschCdlChannel
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings, UserSettings
from nr_pusch.noise import add_awgn, add_awgn_resource_grid
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class MimoCdlChannelTest(unittest.TestCase):
    def test_multiantenna_cdl_keeps_physical_tx_and_rx_axes(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        channel_base = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        channel_settings = replace(channel_base, antennas=replace(
            channel_base.antennas, tx_num_rows=2, tx_num_cols=2, rx_num_rows=2,
            rx_num_cols=4))
        for users in (1, 2):
            with self.subTest(users=users):
                settings = replace(
                    base,
                    pusch=replace(base.pusch, waveform="cp_ofdm", num_layers=4,
                                  num_antenna_ports=4, dmrs_length=2 if users == 2 else 1,
                                  n_size_bwp=6, mcs_index=4),
                    users=tuple(UserSettings(f"ue{i}", i + 1,
                                             tuple(range(i * 4, i * 4 + 4)))
                                for i in range(users)),
                )
                tx = NrPuschTx(settings, device="cpu")
                sent = tx.generate(seed=users)
                sionna.phy.config.seed = users
                channel = NrPuschCdlChannel(channel_settings, device="cpu")
                frequency = channel.apply_frequency(sent.frequency_grid, tx._tx_freq.resource_grid)
                self.assertEqual(frequency.grid.shape[:3], (1, 1, 8))
                self.assertEqual(frequency.channel_frequency_response.shape[:5],
                                 (1, 1, 8, users, 4))
                torch.testing.assert_close(frequency.grid[:, 0], frequency.per_user_grid.sum(1))
                noisy_grid = add_awgn_resource_grid(frequency.grid, 50, seed=7)
                self.assertEqual(noisy_grid.noise_variance.shape, (1, 1, 8))
                time = channel.apply(sent.iq, sent.sample_rate_hz)
                self.assertEqual(time.channel_taps.shape[:4], (1, users, 8, 4))
                torch.testing.assert_close(time.iq, time.per_user_iq.sum(1))
                self.assertEqual(add_awgn(time.iq, 50, seed=7).noise_variance.shape,
                                 (1, 1, 8))


if __name__ == "__main__":
    unittest.main()
