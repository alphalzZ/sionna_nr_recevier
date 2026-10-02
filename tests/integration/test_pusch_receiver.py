from dataclasses import replace
from pathlib import Path
import unittest

import torch

from nr_pusch.channel import NrPuschCdlChannel
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.noise import add_awgn
from nr_pusch.receiver import DftSOfdmDmrsEstimator, NrPuschRx
from nr_pusch.transmitter import NrPuschTx
import sionna.phy


ROOT = Path(__file__).parents[2]


class PuschReceiverTest(unittest.TestCase):
    def test_additional_dmrs_positions_decode_with_dmrs_estimator(self):
        sionna.phy.config.seed = 31
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        channel_settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")

        for additional_position in (1, 2):
            with self.subTest(additional_position=additional_position):
                variant = replace(
                    settings,
                    pusch=replace(
                        settings.pusch,
                        dmrs_additional_position=additional_position,
                    ),
                )
                tx_result = NrPuschTx(variant, device="cpu").generate(
                    batch_size=1,
                    seed=31,
                )
                channel_result = NrPuschCdlChannel(
                    channel_settings,
                    device="cpu",
                ).apply(tx_result.iq, tx_result.sample_rate_hz)
                noisy = add_awgn(channel_result.iq, snr_db=60.0, seed=31)
                rx = NrPuschRx(
                    variant,
                    channel_estimator="dmrs",
                    max_delay_spread_s=channel_settings.channel.max_delay_spread_s,
                    device="cpu",
                )

                result = rx.receive(noisy.iq, noisy.noise_variance)

                self.assertTrue(torch.all(result.crc_status).item())
                torch.testing.assert_close(result.bits, tx_result.bits, rtol=0, atol=0)

    def test_additional_dmrs_symbols_interpolate_time_varying_channel(self):
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        for additional_position, expected_dmrs_count in ((1, 2), (2, 3)):
            with self.subTest(additional_position=additional_position):
                variant = replace(
                    settings,
                    pusch=replace(
                        settings.pusch,
                        dmrs_additional_position=additional_position,
                    ),
                )
                tx = NrPuschTx(variant, device="cpu")
                tx_result = tx.generate(batch_size=1, seed=37)
                tx_grid = tx_result.frequency_grid[0, :, 0]
                mask = tx._tx_freq.pilot_pattern.mask[0, 0]
                dmrs_symbols = tuple(
                    int(symbol) for symbol in torch.where(mask.any(dim=-1))[0].tolist()
                )
                self.assertEqual(len(dmrs_symbols), expected_dmrs_count)

                estimator = DftSOfdmDmrsEstimator(
                    tx_grid[:, dmrs_symbols[0], :],
                    tx._tx_freq.resource_grid,
                    l_min=0,
                    l_max=0,
                )
                num_symbols, num_subcarriers = tx_grid.shape[-2:]
                symbol_time = torch.arange(num_symbols, dtype=torch.float32)
                channel_start = torch.tensor(
                    [0.8 + 0.1j, 0.9 - 0.2j, 1.1 + 0.15j, 0.7 - 0.1j],
                    dtype=torch.complex64,
                )
                channel_slope = torch.tensor(
                    [0.01 - 0.005j, -0.008 + 0.004j, 0.006 + 0.003j, -0.004 - 0.006j],
                    dtype=torch.complex64,
                )
                rx_gain = torch.tensor(
                    [1.0 + 0.0j, 0.8 + 0.2j, 0.7 - 0.1j, 0.9 + 0.15j],
                    dtype=torch.complex64,
                )
                channel = (
                    channel_start[:, None, None]
                    + channel_slope[:, None, None] * symbol_time[None, None, :]
                ) * rx_gain[None, :, None]
                received = torch.zeros(
                    (1, 1, 4, num_symbols, num_subcarriers),
                    dtype=tx_grid.dtype,
                )
                for symbol in dmrs_symbols:
                    for rx_ant in range(4):
                        received[0, 0, rx_ant, symbol] = (
                            tx_grid[:, symbol, :]
                            * channel[:, rx_ant, symbol, None]
                        ).sum(dim=0)

                h_hat, err_var = estimator(
                    received,
                    torch.zeros((1, 1, 4), dtype=torch.float32),
                )

                for symbol in range(dmrs_symbols[0], dmrs_symbols[-1] + 1):
                    torch.testing.assert_close(
                        h_hat[0, 0, :, :, 0, symbol, 0],
                        channel[:, :, symbol].transpose(0, 1),
                        rtol=1e-4,
                        atol=1e-5,
                    )
                torch.testing.assert_close(
                    h_hat[0, 0, :, :, 0, dmrs_symbols[0] - 1, 0],
                    channel[:, :, dmrs_symbols[0]].transpose(0, 1),
                    rtol=1e-4,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    h_hat[0, 0, :, :, 0, dmrs_symbols[-1] + 1, 0],
                    channel[:, :, dmrs_symbols[-1]].transpose(0, 1),
                    rtol=1e-4,
                    atol=1e-5,
                )
                self.assertTrue(torch.all(err_var == 0).item())

                positive_noise = torch.full((1, 1, 4), 0.01, dtype=torch.float32)
                _, noisy_err_var = estimator(received, positive_noise)
                left_symbol, right_symbol = dmrs_symbols[:2]
                middle_symbol = (left_symbol + right_symbol) // 2
                right_weight = (middle_symbol - left_symbol) / (right_symbol - left_symbol)
                expected_error = (
                    noisy_err_var[0, 0, :, :, 0, left_symbol, 0] * (1.0 - right_weight) ** 2
                    + noisy_err_var[0, 0, :, :, 0, right_symbol, 0] * right_weight**2
                )
                torch.testing.assert_close(
                    noisy_err_var[0, 0, :, :, 0, middle_symbol, 0],
                    expected_error,
                    rtol=1e-5,
                    atol=1e-7,
                )

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
