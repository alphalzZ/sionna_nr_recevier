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

        # Once time IQ has been OFDM-demodulated, both receiver input modes
        # must use exactly the same frequency-domain DMRS and detector chain.
        demodulated_grid = dmrs_rx._receiver._ofdm_demodulator(noisy.iq.unsqueeze(1))
        grid_rx = NrPuschRx(
            settings,
            channel_estimator="dmrs",
            max_delay_spread_s=channel_settings.channel.max_delay_spread_s,
            estimate_delay=True,
            input_domain="frequency",
            device="cpu",
        )
        time_h, time_err = dmrs_rx._receiver._channel_estimator(
            demodulated_grid, noisy.noise_variance
        )
        grid_h, grid_err = grid_rx._receiver._channel_estimator(
            demodulated_grid, noisy.noise_variance
        )
        torch.testing.assert_close(time_h, grid_h, rtol=0, atol=0)

        torch.testing.assert_close(time_err, grid_err, rtol=0, atol=0)
        grid_result = grid_rx.receive_frequency_grid(demodulated_grid, noisy.noise_variance)
        torch.testing.assert_close(grid_result.bits, dmrs_result.bits, rtol=0, atol=0)
        self.assertTrue(torch.equal(grid_result.crc_status, dmrs_result.crc_status))
        self.assertEqual(
            set(grid_result.metadata["estimated_bulk_delay_samples_by_user"]),
            {user.name for user in settings.users},
        )

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

        pic_rx = NrPuschRx(
            settings,
            channel_estimator="dmrs",
            detector="mmse-pic",
            max_delay_spread_s=channel_settings.channel.max_delay_spread_s,
            device="cpu",
        )
        pic_result = pic_rx.receive(noisy.iq, noisy.noise_variance)
        self.assertTrue(torch.all(pic_result.crc_status).item())
        torch.testing.assert_close(pic_result.bits, tx_result.bits, rtol=0, atol=0)

        soft_pic_rx = NrPuschRx(
            settings,
            channel_estimator="perfect",
            detector="soft-mmse-pic",
            detector_parameter=1,
            detector_damping=0.25,
            max_delay_spread_s=channel_settings.channel.max_delay_spread_s,
            device="cpu",
        )
        soft_pic_result = soft_pic_rx.receive(
            noisy.iq,
            noisy.noise_variance,
            channel_taps=channel_result.channel_taps,
        )
        self.assertTrue(torch.all(soft_pic_result.crc_status).item())
        torch.testing.assert_close(soft_pic_result.bits, tx_result.bits, rtol=0, atol=0)
        self.assertEqual(soft_pic_result.metadata["detector_feedback_iterations"], 1)
        self.assertEqual(soft_pic_result.metadata["detector_damping"], 0.25)



if __name__ == "__main__":
    unittest.main()
