import dataclasses
import tempfile
import tomllib
import unittest
from pathlib import Path

from nr_pusch.simulation_config import BlerSettings


ROOT = Path(__file__).parents[2]


class BlerSettingsTest(unittest.TestCase):
    def test_reads_detector_comparison_and_cpu_device(self):
        settings = BlerSettings.from_toml(ROOT / "configs" / "bler_4ue_cdl.toml")
        self.assertEqual(settings.detectors, ("lmmse", "lmmse-sic", "k-best", "mmse-pic", "ep", "soft-mmse-pic"))
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.channel_domain, "frequency")
        self.assertIsNone(settings.detector_parameter)
        self.assertEqual(settings.detector_parameters, {"k-best": 16, "mmse-pic": 4, "ep": 10, "soft-mmse-pic": 1})

    def test_estimator_matrix_profile_selects_requested_estimators_and_detector(self):
        settings = BlerSettings.from_toml(ROOT / "configs" / "bler_estimator_matrix.toml")
        self.assertEqual(
            settings.channel_estimators_for_sweep,
            ("dmrs", "dmrs-lmmse", "perfect"),
        )
        self.assertEqual(settings.detectors, ("k-best",))
        self.assertEqual(settings.detector_parameters, {"k-best": 16})

    def test_soft_mmse_pic_runs_with_each_profile_batch_size(self):
        for name, batch_size, feedback_rounds in (
            ("bler_4ue_cdl.toml", 2, 1),
            ("bler_4ue_cdl_gpu.toml", 8, 4),
        ):
            settings = BlerSettings.from_toml(ROOT / "configs" / name)
            with self.subTest(config=name):
                self.assertIn("soft-mmse-pic", settings.detectors)
                self.assertEqual(settings.detector_parameters["soft-mmse-pic"], feedback_rounds)
                self.assertEqual(settings.detector_damping, 0.25)
                self.assertEqual(settings.batch_size_for_detector("soft-mmse-pic"), batch_size)
                self.assertEqual(settings.detector, "lmmse-sic")

    def test_shipped_sweep_keeps_its_own_detector_baseline(self):
        """The capture tuning must not be copied in: the paired sweep measured a regression."""
        for name in ("bler_4ue_cdl.toml", "bler_4ue_cdl_gpu.toml"):
            settings = BlerSettings.from_toml(ROOT / "configs" / name)
            with self.subTest(config=name):
                self.assertEqual(settings.detector_parameters["mmse-pic"], 4)
                self.assertEqual(settings.detector_damping, 0.25)
                self.assertEqual(settings.l_min, -6)
                # Unset means "follow the CDL channel", which is the tuned baseline.
                self.assertIsNone(settings.max_delay_spread_s)

    def test_capture_detector_config_is_reproducible_from_a_sweep(self):
        """The receiver window and detector knobs can carry a capture profile verbatim."""
        with (ROOT / "configs" / "rx_pusch_4ue.toml").open("rb") as stream:
            receiver = tomllib.load(stream)["receiver"]
        settings = BlerSettings.from_toml(ROOT / "configs" / "bler_4ue_cdl.toml")
        aligned = dataclasses.replace(
            settings,
            detector_parameters={"mmse-pic": receiver["detector_parameter"]},
            detector_damping=receiver["detector_damping"],
            l_min=receiver.get("l_min", -6),
            max_delay_spread_s=receiver["max_delay_spread_s"],
        )
        aligned.validate()
        self.assertEqual(aligned.detector_parameters["mmse-pic"], receiver["detector_parameter"])
        self.assertEqual(aligned.detector_damping, receiver["detector_damping"])
        self.assertEqual(aligned.l_min, -6)
        self.assertEqual(aligned.max_delay_spread_s, receiver["max_delay_spread_s"])

    def test_receiver_window_defaults_to_the_channel_value(self):
        settings = BlerSettings.from_toml(ROOT / "configs" / "bler_smoke.toml")
        self.assertIsNone(settings.max_delay_spread_s)
        self.assertEqual(settings.l_min, -6)

    def test_reads_receiver_window_from_toml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bler.toml"
            path.write_text(
                "\n".join(
                    [
                        "[bler]",
                        "snr_db = [10.0]",
                        "batch_size = 1",
                        "max_frames_per_snr = 1",
                        "target_block_errors = 1",
                        "seed = 7",
                        "num_decoder_iterations = 1",
                        'channel_estimator = "dmrs"',
                        'detector = "mmse-pic"',
                        "l_min = -44",
                        "max_delay_spread_s = 2e-6",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            settings = BlerSettings.from_toml(path)
        self.assertEqual(settings.l_min, -44)
        self.assertEqual(settings.max_delay_spread_s, 2e-6)

    def test_lmmse_profile_needs_no_prior_path_and_rejects_removed_key(self):
        common = [
            "[bler]",
            "snr_db = [25.0]",
            "batch_size = 1",
            "max_frames_per_snr = 1",
            "target_block_errors = 1",
            "seed = 1",
            "num_decoder_iterations = 1",
            'channel_estimator = "dmrs-lmmse"',
            'detector = "soft-mmse-pic"',
        ]
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "validation.toml"
            config_path.write_text("\n".join(common), encoding="utf-8")
            settings = BlerSettings.from_toml(config_path)
            self.assertEqual(settings.channel_estimators_for_sweep, ("dmrs-lmmse",))
            self.assertFalse(hasattr(settings, "dmrs_tap_power_prior_path"))
            config_path.write_text(
                "\n".join(common + ['dmrs_tap_power_prior_path = "artifacts/prior.npz"']),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "dmrs_tap_power_prior_path 已移除"):
                BlerSettings.from_toml(config_path)


    def test_rejects_invalid_receiver_window(self):
        common = dict(
            snr_db=(10.0,), batch_size=1, max_frames_per_snr=1, target_block_errors=1,
            seed=1, num_decoder_iterations=1, channel_estimator="dmrs",
            detector="lmmse", detector_parameter=None, detector_parameters={},
            detector_damping=0.25, detectors=("lmmse",), device="cpu",
            channel_domain="frequency",
        )
        for kwargs in ({"l_min": 1.5}, {"l_min": True}, {"max_delay_spread_s": 0.0},
                       {"max_delay_spread_s": -1e-6}, {"max_delay_spread_s": float("nan")}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    BlerSettings(**common, **kwargs).validate()

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
