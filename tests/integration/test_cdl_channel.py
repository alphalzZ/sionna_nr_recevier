from pathlib import Path
import unittest

import torch
import sionna.phy
from sionna.phy.channel import time_to_ofdm_channel
from sionna.phy.ofdm import OFDMDemodulator

from nr_pusch.channel import NrPuschCdlChannel
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.noise import add_awgn_resource_grid
from nr_pusch.receiver import NrPuschRx
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class CdlChannelTest(unittest.TestCase):
    def test_applies_uplink_cdl_to_four_user_waveform(self):
        tx_settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        tx = NrPuschTx(tx_settings, device="cpu")
        tx_result = tx.generate(batch_size=1, seed=7)
        settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        sionna.phy.config.seed = 7
        torch.manual_seed(7)
        channel = NrPuschCdlChannel(settings, device="cpu")

        result = channel.apply(tx_result.iq, tx_result.sample_rate_hz)

        self.assertEqual(result.iq.ndim, 3)
        self.assertEqual(tuple(result.iq.shape[:2]), (1, 4))
        self.assertEqual(tuple(result.per_user_iq.shape[:3]), (1, 4, 4))
        self.assertEqual(tuple(result.channel_taps.shape[:3]), (1, 4, 4))
        self.assertEqual(result.iq.shape[-1], tx_result.iq.shape[-1] + result.metadata["channel_tail_samples"])
        self.assertTrue(torch.isfinite(result.iq).all())
        torch.testing.assert_close(result.iq, result.per_user_iq.sum(dim=1))

        # With zero UE speed, the same CDL realization must yield the same
        # channel coefficients in both domains. The remaining received-grid
        # error is the small OFDM boundary effect of the time convolution.
        sionna.phy.config.seed = 7
        torch.manual_seed(7)
        frequency = NrPuschCdlChannel(settings, device="cpu").apply_frequency(
            tx_result.frequency_grid, tx._tx_freq.resource_grid
        )
        rg = tx._tx_freq.resource_grid
        l_min = result.metadata["time_lag_min"]
        time_grid = OFDMDemodulator(
            rg.fft_size, l_min, rg.cyclic_prefix_length, device="cpu"
        )(result.iq.unsqueeze(1))
        relative_error = (
            (time_grid - frequency.grid).abs().square().mean()
            / time_grid.abs().square().mean()
        ).sqrt()
        self.assertLess(relative_error.item(), 0.01)
        time_taps = result.channel_taps.permute(0, 2, 1, 3, 4).unsqueeze(1).unsqueeze(4)
        time_csi = time_to_ofdm_channel(time_taps, rg, l_min)
        torch.testing.assert_close(
            time_csi, frequency.channel_frequency_response, rtol=1e-4, atol=1e-4
        )

    def test_rejects_wrong_transmit_iq_axes(self):
        settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        channel = NrPuschCdlChannel(settings, device="cpu")
        with self.assertRaisesRegex(ValueError, "形状必须为"):
            channel.apply(torch.ones((1, 4, 4, 128), dtype=torch.complex64), 18_000_000)

    def test_frequency_domain_channel_decodes_batched_dft_s_ofdm_grid(self):
        tx_settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        tx = NrPuschTx(tx_settings, device="cpu")
        tx_result = tx.generate(batch_size=2, seed=29)
        settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        channel_result = NrPuschCdlChannel(settings, device="cpu").apply_frequency(
            tx_result.frequency_grid,
            tx._tx_freq.resource_grid,
        )

        self.assertEqual(tuple(channel_result.grid.shape), (2, 1, 4, 14, 600))
        self.assertEqual(
            tuple(channel_result.channel_frequency_response.shape), (2, 1, 4, 4, 1, 14, 600)
        )
        torch.testing.assert_close(
            channel_result.grid[:, 0], channel_result.per_user_grid.sum(dim=1)
        )
        noisy = add_awgn_resource_grid(channel_result.grid, snr_db=60.0, seed=29)
        rx = NrPuschRx(
            tx_settings,
            channel_estimator="perfect",
            detector="lmmse",
            input_domain="frequency",
            device="cpu",
        )
        result = rx.receive_frequency_grid(
            noisy.grid,
            noisy.noise_variance,
            channel_frequency_response=channel_result.channel_frequency_response,
        )
        self.assertTrue(torch.all(result.crc_status).item())
        torch.testing.assert_close(result.bits, tx_result.bits, rtol=0, atol=0)

        dmrs_rx = NrPuschRx(
            tx_settings,
            channel_estimator="dmrs",
            detector="lmmse",
            input_domain="frequency",
            device="cpu",
        )
        dmrs_result = dmrs_rx.receive_frequency_grid(noisy.grid, noisy.noise_variance)
        self.assertEqual(tuple(dmrs_result.bits.shape), tuple(tx_result.bits.shape))
        self.assertTrue(torch.all(dmrs_result.crc_status).item())
        torch.testing.assert_close(dmrs_result.bits, tx_result.bits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
