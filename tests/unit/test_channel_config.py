from pathlib import Path
import tempfile
import unittest

from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings


ROOT = Path(__file__).parents[2]


class ChannelSettingsTest(unittest.TestCase):
    def test_loads_four_receive_antenna_cdl_profile(self):
        settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")

        self.assertEqual(settings.channel.model, "A")
        self.assertEqual(settings.channel.direction, "uplink")
        self.assertEqual(settings.antennas.rx_num_rows * settings.antennas.rx_num_cols, 4)

    def test_accepts_arbitrary_positive_receive_array(self):
        source = (ROOT / "configs" / "cdl_38_901_4x4.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "eight_rx.toml"
            path.write_text(source.replace("rx_num_cols = 2", "rx_num_cols = 4"), encoding="utf-8")
            self.assertEqual(ChannelSettings.from_toml(path).antennas.rx_num_cols, 4)

    def test_rejects_nonpositive_array_dimension(self):
        source = (ROOT / "configs" / "cdl_38_901_4x4.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.toml"
            path.write_text(source.replace("rx_num_cols = 2", "rx_num_cols = 0"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "正整数"):
                ChannelSettings.from_toml(path)

    def test_transmitter_ports_must_match_ut_array(self):
        settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")
        tx = TxSettings.from_toml(ROOT / "configs" / "pusch_1ue_4layer.toml")
        with self.assertRaisesRegex(ValueError, "num_antenna_ports"):
            settings.validate_transmitter(tx)
        matching = ChannelSettings.from_toml(
            ROOT / "configs" / "cdl_38_901_4tx_8rx.toml")
        matching.validate_transmitter(tx)
        self.assertEqual(matching.antennas.rx_num_rows * matching.antennas.rx_num_cols, 8)


if __name__ == "__main__":
    unittest.main()
