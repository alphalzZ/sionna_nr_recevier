from pathlib import Path
import unittest

from nr_pusch.simulation_config import BlerSettings


ROOT = Path(__file__).parents[2]


class BlerSettingsTest(unittest.TestCase):
    def test_reads_detector_comparison_and_cpu_device(self):
        settings = BlerSettings.from_toml(ROOT / "configs" / "bler_4ue_cdl.toml")
        self.assertEqual(settings.detectors, ("lmmse", "lmmse-sic", "k-best", "mmse-pic", "ep"))
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.channel_domain, "frequency")
        self.assertIsNone(settings.detector_parameter)
        self.assertEqual(settings.detector_parameters, {"k-best": 16, "mmse-pic": 4, "ep": 10})
        self.assertEqual(settings.detector_damping, 0.25)

    def test_rejects_unknown_detector(self):
        settings = BlerSettings(
            snr_db=(10.0,), batch_size=1, max_frames_per_snr=1,
            target_block_errors=1, seed=1, num_decoder_iterations=1,
            channel_estimator="dmrs", detector="unknown", detector_parameter=None,
            detector_parameters={},
            detector_damping=0.25,
            detectors=("unknown",), device="cpu", channel_domain="frequency",
        )
        with self.assertRaises(ValueError):
            settings.validate()


if __name__ == "__main__":
    unittest.main()
