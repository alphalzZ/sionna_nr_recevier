from pathlib import Path
import unittest

import torch
import sionna.phy

from nr_pusch.bler import estimate_dmrs_tap_power_prior
from nr_pusch.channel import NrPuschCdlChannel
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.noise import add_awgn_resource_grid
from nr_pusch.receiver import NrPuschRx
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class DmrsLmmseIntegrationTest(unittest.TestCase):
    def test_lmmse_uses_same_high_snr_frame_and_returns_valid_csi(self):
        tx_settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        channel_settings = ChannelSettings.from_toml(
            ROOT / "configs" / "cdl_38_901_4x4.toml"
        )
        prior = estimate_dmrs_tap_power_prior(
            tx_settings,
            channel_settings,
            l_min=-6,
            max_delay_spread_s=3e-6,
            num_realizations=16,
            seed=21260924,
            device="cpu",
        )
        sionna.phy.config.seed = 13
        torch.manual_seed(13)
        transmitter = NrPuschTx(tx_settings, device="cpu")
        tx_result = transmitter.generate(batch_size=1, seed=13)
        channel_result = NrPuschCdlChannel(channel_settings, device="cpu").apply_frequency(
            tx_result.frequency_grid, transmitter._tx_freq.resource_grid
        )
        noisy = add_awgn_resource_grid(channel_result.grid, 60.0, seed=13)
        truth = channel_result.channel_frequency_response
        pilot_mask = transmitter._tx_freq.pilot_pattern.mask[0, 0].bool()
        data_mask = (tx_result.frequency_grid[0, :, 0].abs() > 0) & ~pilot_mask[None, :, :]

        for estimator in ("dmrs", "dmrs-lmmse", "perfect"):
            receiver = NrPuschRx(
                tx_settings,
                channel_estimator=estimator,
                dmrs_tap_power_prior=prior if estimator == "dmrs-lmmse" else None,
                max_delay_spread_s=3e-6,
                num_decoder_iterations=20,
                detector="soft-mmse-pic",
                detector_parameter=1,
                detector_damping=0.25,
                input_domain="frequency",
                device="cpu",
            )
            if estimator == "perfect":
                h_hat = truth
                err_var = torch.zeros_like(truth.real)
            else:
                h_hat, err_var = receiver._estimator(noisy.grid, noisy.noise_variance)
            self.assertEqual(tuple(h_hat.shape), tuple(truth.shape))
            self.assertTrue(torch.isfinite(h_hat).all().item())
            self.assertTrue(torch.isfinite(err_var).all().item())
            self.assertTrue(torch.all(err_var >= 0).item())
            for user in range(4):
                for rx_ant in range(4):
                    selection = data_mask[user]
                    estimated_data = h_hat[0, 0, rx_ant, user, 0][selection]
                    true_data = truth[0, 0, rx_ant, user, 0][selection]
                    nmse = (estimated_data - true_data).abs().square().sum() / true_data.abs().square().sum()
                    self.assertTrue(torch.isfinite(nmse).item())

            result = receiver.receive_frequency_grid(
                noisy.grid,
                noisy.noise_variance,
                channel_frequency_response=truth if estimator == "perfect" else None,
            )
            self.assertTrue(torch.all(result.crc_status).item())
            torch.testing.assert_close(result.bits, tx_result.bits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
