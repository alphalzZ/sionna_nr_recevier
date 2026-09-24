from pathlib import Path
import unittest

import torch

from nr_pusch.channel import NrPuschCdlChannel
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class CdlChannelTest(unittest.TestCase):
    def test_applies_uplink_cdl_to_four_user_waveform(self):
        tx_settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        tx = NrPuschTx(tx_settings, device="cpu")
        tx_result = tx.generate(batch_size=1, seed=7)
        settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        channel = NrPuschCdlChannel(settings, device="cpu")

        result = channel.apply(tx_result.iq, tx_result.sample_rate_hz)

        self.assertEqual(result.iq.ndim, 3)
        self.assertEqual(tuple(result.iq.shape[:2]), (1, 4))
        self.assertEqual(tuple(result.per_user_iq.shape[:3]), (1, 4, 4))
        self.assertEqual(tuple(result.channel_taps.shape[:3]), (1, 4, 4))
        self.assertEqual(result.iq.shape[-1], tx_result.iq.shape[-1] + result.metadata["channel_tail_samples"])
        self.assertTrue(torch.isfinite(result.iq).all())
        torch.testing.assert_close(result.iq, result.per_user_iq.sum(dim=1))

    def test_rejects_wrong_transmit_iq_axes(self):
        settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        channel = NrPuschCdlChannel(settings, device="cpu")
        with self.assertRaisesRegex(ValueError, "形状必须为"):
            channel.apply(torch.ones((1, 4, 4, 128), dtype=torch.complex64), 18_000_000)


if __name__ == "__main__":
    unittest.main()
