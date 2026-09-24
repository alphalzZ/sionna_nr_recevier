from pathlib import Path
import tempfile
import unittest

from nr_pusch.channel_config import ChannelSettings


ROOT = Path(__file__).parents[2]


class ChannelSettingsTest(unittest.TestCase):
    def test_loads_four_receive_antenna_cdl_profile(self):
        settings = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml")

        self.assertEqual(settings.channel.model, "A")
        self.assertEqual(settings.channel.direction, "uplink")
        self.assertEqual(settings.antennas.rx_num_rows * settings.antennas.rx_num_cols, 4)

    def test_rejects_receiver_array_that_is_not_four_antennas(self):
        source = (ROOT / "configs" / "cdl_38_901_4x4.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.toml"
            path.write_text(source.replace("rx_num_cols = 2", "rx_num_cols = 3"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "恰好包含 4"):
                ChannelSettings.from_toml(path)


if __name__ == "__main__":
    unittest.main()
