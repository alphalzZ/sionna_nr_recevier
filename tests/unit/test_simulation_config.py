from pathlib import Path
import tempfile
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

    def test_stop_at_zero_bler_defaults_to_disabled(self):
        settings = BlerSettings.from_toml(ROOT / "configs" / "bler_smoke.toml")
        self.assertFalse(settings.stop_at_zero_bler)

    def test_reads_stop_at_zero_bler_from_toml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bler.toml"
            path.write_text(
                "\n".join(
                    [
                        "[bler]",
                        "snr_db = [10.0, 20.0]",
                        "batch_size = 1",
                        "max_frames_per_snr = 1",
                        "target_block_errors = 1",
                        "seed = 7",
                        "num_decoder_iterations = 1",
                        'channel_estimator = "dmrs"',
                        'detector = "lmmse"',
                        "stop_at_zero_bler = true",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            settings = BlerSettings.from_toml(path)
        self.assertTrue(settings.stop_at_zero_bler)

    def test_rejects_non_boolean_stop_at_zero_bler(self):
        settings = BlerSettings(
            snr_db=(10.0,), batch_size=1, max_frames_per_snr=1,
            target_block_errors=1, seed=1, num_decoder_iterations=1,
            channel_estimator="dmrs", detector="lmmse", detector_parameter=None,
            detector_parameters={}, detector_damping=0.25,
            detectors=("lmmse",), device="cpu", channel_domain="frequency",
            stop_at_zero_bler="yes",
        )
        with self.assertRaises(ValueError):
            settings.validate()

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
