from dataclasses import replace
from pathlib import Path
import unittest

from nr_pusch.bler import simulate_bler
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings, UserSettings
from nr_pusch.simulation_config import BlerSettings


ROOT = Path(__file__).parents[2]


class MimoBlerTest(unittest.TestCase):
    def test_eight_stream_dmrs_sweep_counts_two_transport_blocks(self):
        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        tx_settings = replace(
            base, pusch=replace(base.pusch, num_layers=4, num_antenna_ports=4,
                                dmrs_length=2, n_size_bwp=12, mcs_index=8),
            users=(UserSettings("ue0", 1, (0, 1, 2, 3)),
                   UserSettings("ue1", 2, (4, 5, 6, 7))),
        )
        channel = ChannelSettings.from_toml(
            ROOT / "configs" / "cdl_38_901_4tx_8rx.toml")
        smoke = BlerSettings.from_toml(ROOT / "configs" / "bler_smoke.toml")
        simulation = replace(smoke, snr_db=(65.0,), max_frames_per_snr=1,
                             target_block_errors=1, l_min=-2,
                             max_delay_spread_s=0.3e-6)
        sweep = simulate_bler(tx_settings, channel, simulation, device="cpu")
        point = sweep.points[0]
        self.assertEqual((point.frames, point.transport_blocks), (1, 2))
        self.assertEqual((point.block_errors, point.crc_failures, point.bit_errors),
                         (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
