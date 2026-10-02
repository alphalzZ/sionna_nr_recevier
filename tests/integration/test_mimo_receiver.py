from dataclasses import replace
from pathlib import Path
import unittest

import torch
from sionna.phy.channel import time_lag_discrete_time_channel
import sionna.phy

from nr_pusch.channel import NrPuschCdlChannel
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings, UserSettings
from nr_pusch.noise import add_awgn, add_awgn_resource_grid
from nr_pusch.receiver import NrPuschRx
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class MimoReceiverTest(unittest.TestCase):
    def test_perfect_and_dmrs_csi_restore_multi_layer_transport_blocks(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        channel_base = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        for users, layers, waveform, precoding in (
            (1, 1, "dft_s_ofdm", "non-codebook"),
            (1, 4, "dft_s_ofdm", "non-codebook"),
            (2, 4, "dft_s_ofdm", "non-codebook"),
            (1, 3, "dft_s_ofdm", "codebook"),
            (1, 3, "cp_ofdm", "codebook"),
        ):
            with self.subTest(users=users, layers=layers, waveform=waveform):
                ports = 4 if layers >= 3 else layers
                settings = replace(
                    base,
                    pusch=replace(base.pusch, waveform=waveform, num_layers=layers,
                                  num_antenna_ports=ports, precoding=precoding,
                                  dmrs_length=2 if users * layers > 4 else 1,
                                  n_size_bwp=12, mcs_index=8),
                    users=tuple(UserSettings(f"ue{i}", i + 1,
                                             tuple(range(i * layers, (i + 1) * layers)))
                                for i in range(users)),
                )
                channel_settings = replace(channel_base, antennas=replace(
                    channel_base.antennas, tx_num_rows=1, tx_num_cols=ports,
                    rx_num_rows=2, rx_num_cols=4 if layers > 1 else 2))
                sionna.phy.config.seed = 4
                tx = NrPuschTx(settings, device="cpu")
                sent = tx.generate(seed=4)
                received = NrPuschCdlChannel(channel_settings, device="cpu").apply_frequency(
                    sent.frequency_grid, tx._tx_freq.resource_grid)
                noisy = add_awgn_resource_grid(received.grid, 65, seed=4)
                for estimator in ("perfect", "dmrs"):
                    with self.subTest(estimator=estimator):
                        rx = NrPuschRx(settings, channel_estimator=estimator,
                                       input_domain="frequency", detector="lmmse",
                                       l_min=-2, max_delay_spread_s=0.3e-6, device="cpu")
                        decoded = rx.receive_frequency_grid(
                            noisy.grid, noisy.noise_variance,
                            channel_frequency_response=(received.channel_frequency_response
                                                        if estimator == "perfect" else None))
                        self.assertTrue(torch.all(decoded.crc_status).item())
                        torch.testing.assert_close(decoded.bits, sent.bits, rtol=0, atol=0)
                        self.assertEqual(decoded.constellation.shape[:3], (1, users, layers))


    def test_four_ue_single_layer_codebook_dmrs_decodes(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        settings = replace(
            base,
            pusch=replace(
                base.pusch, num_layers=1, num_antenna_ports=2,
                precoding="codebook", dmrs_length=1, n_size_bwp=12,
                mcs_index=8,
            ),
            users=tuple(
                UserSettings(f"ue{i}", i + 1, (i,)) for i in range(4)
            ),
        )
        channel_base = ChannelSettings.from_toml(
            ROOT / "configs" / "cdl_38_901_4x4.toml"
        )
        channel_settings = replace(
            channel_base,
            antennas=replace(
                channel_base.antennas,
                tx_num_rows=1, tx_num_cols=2,
                rx_num_rows=2, rx_num_cols=2,
            ),
        )
        sionna.phy.config.seed = 4
        tx = NrPuschTx(settings, device="cpu")
        sent = tx.generate(seed=4)
        channel = NrPuschCdlChannel(
            channel_settings, device="cpu"
        ).apply_frequency(sent.frequency_grid, tx._tx_freq.resource_grid)
        noisy = add_awgn_resource_grid(channel.grid, 65, seed=4)
        rx = NrPuschRx(
            settings, channel_estimator="dmrs", input_domain="frequency",
            device="cpu",
        )

        decoded = rx.receive_frequency_grid(noisy.grid, noisy.noise_variance)

        self.assertTrue(torch.all(decoded.crc_status).item())
        torch.testing.assert_close(decoded.bits, sent.bits, rtol=0, atol=0)
        self.assertEqual(decoded.constellation.shape[:3], (1, 4, 1))


    def test_multilayer_dmrs_lmmse_uses_tap_prior(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        settings = replace(base, pusch=replace(
            base.pusch, num_layers=4, num_antenna_ports=4, n_size_bwp=12, mcs_index=8),
            users=(UserSettings("ue0", 1, (0, 1, 2, 3)),))
        channel_base = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        channel_settings = replace(channel_base, antennas=replace(
            channel_base.antennas, tx_num_cols=4, rx_num_rows=2, rx_num_cols=4))
        sionna.phy.config.seed = 4
        tx = NrPuschTx(settings, device="cpu")
        sent = tx.generate(seed=4)
        channel = NrPuschCdlChannel(channel_settings, device="cpu").apply_frequency(
            sent.frequency_grid, tx._tx_freq.resource_grid)
        noisy = add_awgn_resource_grid(channel.grid, 65, seed=4)
        _, l_max = time_lag_discrete_time_channel(tx.sample_rate_hz, 0.3e-6)
        rx = NrPuschRx(settings, channel_estimator="dmrs-lmmse",
                       dmrs_tap_power_prior=torch.ones(l_max + 3), l_min=-2,
                       max_delay_spread_s=0.3e-6, input_domain="frequency", device="cpu")
        decoded = rx.receive_frequency_grid(noisy.grid, noisy.noise_variance)
        self.assertTrue(torch.all(decoded.crc_status).item())
        torch.testing.assert_close(decoded.bits, sent.bits, rtol=0, atol=0)

    def test_single_tx_four_rx_time_iq_perfect_and_dmrs(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        settings = replace(base, pusch=replace(base.pusch, n_size_bwp=6, mcs_index=8),
                           users=(UserSettings("ue0", 1, (0,)),))
        channel_settings = ChannelSettings.from_toml(
            ROOT / "configs" / "cdl_38_901_4x4.toml"
        )
        sionna.phy.config.seed = 13
        sent = NrPuschTx(settings, device="cpu").generate(seed=13)
        received = NrPuschCdlChannel(channel_settings, device="cpu").apply(
            sent.iq, sent.sample_rate_hz)
        noisy = add_awgn(received.iq, 70, seed=13)
        for estimator in ("perfect", "dmrs"):
            with self.subTest(estimator=estimator):
                receiver = NrPuschRx(
                    settings, channel_estimator=estimator, input_domain="time",
                    detector="lmmse", l_min=-2, max_delay_spread_s=0.3e-6,
                    device="cpu")
                decoded = receiver.receive(
                    noisy.iq, noisy.noise_variance,
                    channel_taps=received.channel_taps if estimator == "perfect" else None)
                self.assertTrue(torch.all(decoded.crc_status).item())
                torch.testing.assert_close(decoded.bits, sent.bits, rtol=0, atol=0)

    def test_all_six_detectors_restore_two_layer_transport_blocks(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        channel_base = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        channel_settings = replace(channel_base, antennas=replace(
            channel_base.antennas, tx_num_cols=2))
        for users in (1, 2):
            settings = replace(base, pusch=replace(
                base.pusch, num_layers=2, num_antenna_ports=2,
                n_size_bwp=6, mcs_index=8),
                users=tuple(UserSettings(f"ue{i}", i + 1, (2 * i, 2 * i + 1))
                            for i in range(users)))
            sionna.phy.config.seed = 17
            tx = NrPuschTx(settings, device="cpu")
            sent = tx.generate(seed=17)
            channel = NrPuschCdlChannel(channel_settings, device="cpu").apply_frequency(
                sent.frequency_grid, tx._tx_freq.resource_grid)
            noisy = add_awgn_resource_grid(channel.grid, 75, seed=17)
            for detector, parameter in (("lmmse", None), ("mmse-pic", 2),
                                         ("soft-mmse-pic", 1), ("lmmse-sic", None),
                                         ("k-best", 16), ("ep", 4)):
                with self.subTest(detector=detector, users=users):
                    rx = NrPuschRx(settings, channel_estimator="perfect",
                                   input_domain="frequency", detector=detector,
                                   detector_parameter=parameter, device="cpu")
                    decoded = rx.receive_frequency_grid(
                        noisy.grid, noisy.noise_variance,
                        channel_frequency_response=channel.channel_frequency_response)
                    self.assertTrue(torch.all(decoded.crc_status).item())
                    torch.testing.assert_close(decoded.bits, sent.bits, rtol=0, atol=0)
            if users == 2:
                rx = NrPuschRx(settings, channel_estimator="perfect",
                               input_domain="frequency", detector="k-best", device="cpu")
                with self.assertRaisesRegex(ValueError, "接收天线数不少于总流数"):
                    rx.receive_frequency_grid(
                        noisy.grid[:, :, :2], noisy.noise_variance[:, :, :2],
                        channel_frequency_response=channel.channel_frequency_response[:, :, :2],
                    )


if __name__ == "__main__":
    unittest.main()
