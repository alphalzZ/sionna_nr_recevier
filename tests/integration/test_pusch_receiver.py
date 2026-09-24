from pathlib import Path
import unittest

import torch

from nr_pusch.channel import NrPuschCdlChannel
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.noise import add_awgn
from nr_pusch.receiver import NrPuschRx
from nr_pusch.transmitter import NrPuschTx
import sionna.phy


ROOT = Path(__file__).parents[2]


class PuschReceiverTest(unittest.TestCase):
    def test_four_user_dft_s_ofdm_cdl_loopback_passes_crc_and_recovers_payload(self):
        sionna.phy.config.seed = 13
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        tx = NrPuschTx(settings, device="cpu")
        tx_result = tx.generate(batch_size=1, seed=13)
        channel_settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        channel_result = NrPuschCdlChannel(channel_settings, device="cpu").apply(
            tx_result.iq, tx_result.sample_rate_hz
        )
        noisy = add_awgn(channel_result.iq, snr_db=60.0, seed=13)
        rx = NrPuschRx(settings, channel_estimator="perfect", device="cpu")

        result = rx.receive(
            noisy.iq,
            noisy.noise_variance,
            channel_taps=channel_result.channel_taps,
        )

        self.assertTrue(torch.all(result.crc_status).item())
        torch.testing.assert_close(result.bits, tx_result.bits, rtol=0, atol=0)
        self.assertEqual(tuple(result.bits.shape), (1, 4, 28168))

        dmrs_rx = NrPuschRx(
            settings,
            channel_estimator="dmrs",
            max_delay_spread_s=channel_settings.channel.max_delay_spread_s,
            device="cpu",
        )
        dmrs_result = dmrs_rx.receive(noisy.iq, noisy.noise_variance)
        self.assertTrue(torch.all(dmrs_result.crc_status).item())
        torch.testing.assert_close(dmrs_result.bits, tx_result.bits, rtol=0, atol=0)

        sic_rx = NrPuschRx(
            settings,
            channel_estimator="dmrs",
            detector="lmmse-sic",
            max_delay_spread_s=channel_settings.channel.max_delay_spread_s,
            device="cpu",
        )
        sic_result = sic_rx.receive(noisy.iq, noisy.noise_variance)
        self.assertTrue(torch.all(sic_result.crc_status).item())
        torch.testing.assert_close(sic_result.bits, tx_result.bits, rtol=0, atol=0)
        self.assertEqual(set(sic_result.metadata["sic_user_order"]), {u.name for u in settings.users})
        self.assertTrue(all(all(row) for row in sic_result.metadata["sic_crc_before_cancel"]))

        kbest_rx = NrPuschRx(
            settings,
            channel_estimator="dmrs",
            detector="k-best",
            max_delay_spread_s=channel_settings.channel.max_delay_spread_s,
            device="cpu",
        )
        kbest_result = kbest_rx.receive(noisy.iq, noisy.noise_variance)
        self.assertTrue(torch.all(kbest_result.crc_status).item())
        torch.testing.assert_close(kbest_result.bits, tx_result.bits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
