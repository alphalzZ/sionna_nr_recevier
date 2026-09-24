from pathlib import Path
import unittest

from nr_pusch.simulation_config import BlerSettings


ROOT = Path(__file__).parents[2]


class BlerSettingsTest(unittest.TestCase):
    def test_reads_detector_comparison_and_cpu_device(self):
        settings = BlerSettings.from_toml(ROOT / "configs" / "bler_4ue_cdl.toml")
        self.assertEqual(settings.detectors, ("lmmse", "lmmse-sic", "k-best"))
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.detector_parameter, 16)

    def test_rejects_unknown_detector(self):
        settings = BlerSettings(
            snr_db=(10.0,), batch_size=1, max_frames_per_snr=1,
            target_block_errors=1, seed=1, num_decoder_iterations=1,
            channel_estimator="dmrs", detector="unknown", detector_parameter=None,
            detectors=("unknown",), device="cpu",
        )
        with self.assertRaises(ValueError):
            settings.validate()


if __name__ == "__main__":
    unittest.main()
