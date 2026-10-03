from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from nr_pusch.cli import estimator_validation as estimator_validation_cli
from nr_pusch.estimator_validation import (
    _acceptance_gate,
    _boolean_setting,
    _data_re_nmse,
    _paired_bootstrap_upper,
    _positive_override,
    _threshold_setting,
)


class EstimatorValidationCliTest(unittest.TestCase):
    """The published command line must stay runnable end to end."""

    ROOT = Path(__file__).resolve().parents[2]
    TX_CONFIG = ROOT / "configs" / "pusch_4ue.toml"
    CHANNEL_CONFIG = ROOT / "configs" / "cdl_38_901_4x4.toml"
    VALIDATION_CONFIG = ROOT / "configs" / "channel_estimation_validation.toml"

    def _run_main(self, *extra_argv):
        captured = {}

        def fake_validation(*args, **kwargs):
            captured["kwargs"] = kwargs
            return {
                "acceptance_gate": {"passed": True},
                "artifacts": {
                    "summary": "summary.json",
                    "frames": "frames.npz",
                    "prior": "candidate.prior.npz",
                    "published_prior": "published.npz",
                },
            }

        argv = [
            "nr-pusch-estimator-validation",
            "--tx-config", str(self.TX_CONFIG),
            "--channel-config", str(self.CHANNEL_CONFIG),
            "--validation-config", str(self.VALIDATION_CONFIG),
            "--device", "cpu",
            *extra_argv,
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            estimator_validation_cli, "run_estimator_validation", fake_validation
        ):
            estimator_validation_cli.main()
        return captured["kwargs"]

    def test_documented_flags_reach_the_validation_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "validation.json"
            prior_dir = Path(directory) / "store"
            kwargs = self._run_main(
                "--output", str(output),
                "--prior-dir", str(prior_dir),
                "--prior-realizations", "8",
                "--frames-per-snr", "2",
            )

        self.assertEqual(Path(kwargs["output"]), output)
        self.assertEqual(kwargs["prior_dir"], prior_dir)
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["prior_realizations"], 8)
        self.assertEqual(kwargs["frames_per_snr"], 2)

    def test_prior_dir_defaults_to_the_channel_config_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self._run_main(
                "--output", str(Path(directory) / "validation.json")
            )
        self.assertEqual(
            kwargs["prior_dir"], self.CHANNEL_CONFIG.parent / "tap_power_prior"
        )


class EstimatorValidationGateTest(unittest.TestCase):
    def test_data_re_nmse_returns_per_frame_user_and_antenna_values(self):
        truth = torch.ones((2, 1, 2, 4, 2, 3, 5), dtype=torch.complex64)
        estimated = truth.clone()
        estimated[1, 0, 1, 2, 0] *= 2
        data_masks = torch.ones((4, 2, 3, 5), dtype=torch.bool)
        data_masks[:, 1, 0, :] = False

        nmse = _data_re_nmse(estimated, truth, data_masks)

        self.assertEqual(nmse.shape, (2, 4, 2))
        np.testing.assert_array_equal(nmse[0], np.zeros((4, 2)))
        self.assertEqual(nmse[1, 2, 1], 15 / 25)

    def test_paired_bootstrap_resamples_whole_frame_clusters(self):
        difference = np.full(32, -4.0)
        upper = _paired_bootstrap_upper(
            difference, 500, 123, num_users=4
        )
        self.assertEqual(upper, -1.0)

    def test_smoke_overrides_reject_non_positive_values(self):
        config = {"training_realizations": 256}
        self.assertEqual(_positive_override(None, config, "training_realizations", "prior-realizations"), 256)
        self.assertEqual(_positive_override(8, config, "training_realizations", "prior-realizations"), 8)
        for override in (0, -1, True):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "prior-realizations"):
                    _positive_override(
                        override, config, "training_realizations", "prior-realizations"
                    )

    def test_gate_requires_each_snr_threshold_and_high_snr_non_regression(self):
        frame_count = 20
        block_error = np.zeros((3, 3, 3, frame_count, 4), dtype=np.bool_)
        data_re_nmse = np.ones((3, 3, 3, frame_count, 4, 4), dtype=np.float64)
        data_re_nmse[1] = 0.5
        block_error[0, 1, 0:2, :, 0] = True
        block_error[0, 2, 2, :, 0] = True
        accepted = _acceptance_gate(
            block_error, data_re_nmse, (frame_count, frame_count, frame_count), 500, 19
        )
        self.assertTrue(accepted["passed"])

        # A 5% relative reduction at 25 dB must fail even when the paired
        # difference is negative; the 30 dB and 60 dB conditions still pass.
        block_error[1, 1, 0, :19, 0] = True
        rejected = _acceptance_gate(
            block_error, data_re_nmse, (frame_count, frame_count, frame_count), 500, 19
        )
        self.assertFalse(rejected["checks"]["25.0"]["passed"])
        self.assertTrue(rejected["checks"]["30.0"]["passed"])
        self.assertTrue(rejected["checks"]["60.0"]["passed"])
        self.assertFalse(rejected["passed"])

    def _borderline_gate_inputs(self):
        """Holdout data whose 30 dB point sits exactly on the versioned gate."""
        frames = 250
        block_error = np.zeros((3, 3, 3, frames, 4), dtype=np.bool_)
        data_re_nmse = np.ones((3, 3, 3, frames, 4, 4), dtype=np.float64)
        data_re_nmse[1] = 0.5
        for snr_index, (baseline_errors, candidate_errors) in enumerate(
            ((100, 85), (200, 190))
        ):
            # Candidate errors are a subset of baseline errors, so every paired
            # per-frame difference is <= 0 and the 97.5% upper bound is 0.0.
            baseline = block_error[0, 1, snr_index].reshape(-1)
            candidate = block_error[1, 1, snr_index].reshape(-1)
            baseline[:baseline_errors] = True
            candidate[:candidate_errors] = True
        return block_error, data_re_nmse, (frames, frames, frames)

    def test_gate_relaxes_only_through_versioned_thresholds(self):
        block_error, data_re_nmse, counts = self._borderline_gate_inputs()
        pre_registered = _acceptance_gate(block_error, data_re_nmse, counts, 500, 19)
        self.assertFalse(pre_registered["passed"])
        self.assertEqual(pre_registered["thresholds"], {
            "min_relative_bler_reduction": 0.10,
            "max_paired_bler_difference_97_5pct_upper": 0.0,
            "require_strict_upper_bound": True,
            "high_snr_no_regression": True,
        })
        self.assertAlmostEqual(
            pre_registered["checks"]["30.0"]["relative_bler_reduction"], 0.05
        )
        self.assertEqual(
            pre_registered["checks"]["30.0"]["paired_bler_difference_97_5pct_upper"],
            0.0,
        )

        # A non-strict upper bound alone still fails the 5% relative reduction.
        non_strict = _acceptance_gate(
            block_error, data_re_nmse, counts, 500, 19, require_strict_upper_bound=False
        )
        self.assertFalse(non_strict["checks"]["30.0"]["passed"])
        self.assertTrue(non_strict["checks"]["25.0"]["passed"])

        # A lower reduction floor alone still fails the strict upper bound.
        relaxed_floor = _acceptance_gate(
            block_error, data_re_nmse, counts, 500, 19,
            min_relative_bler_reduction=0.04,
        )
        self.assertFalse(relaxed_floor["checks"]["30.0"]["passed"])

        relaxed = _acceptance_gate(
            block_error, data_re_nmse, counts, 500, 19,
            min_relative_bler_reduction=0.04, require_strict_upper_bound=False,
        )
        self.assertTrue(relaxed["checks"]["30.0"]["passed"])
        self.assertTrue(relaxed["passed"])
        self.assertFalse(relaxed["thresholds"]["require_strict_upper_bound"])

    def test_threshold_settings_reject_unusable_values(self):
        self.assertEqual(_threshold_setting({}, "missing_key", 0.09), 0.09)
        self.assertEqual(_boolean_setting({}, "missing_key", False), False)
        for value in (True, -0.01, "0.09", float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "estimator_validation.min_relative_bler_reduction"
                ):
                    _threshold_setting(
                        {"min_relative_bler_reduction": value},
                        "min_relative_bler_reduction",
                        0.10,
                    )
        for value in (0, 1, "true", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "estimator_validation.require_strict_upper_bound"
                ):
                    _boolean_setting(
                        {"require_strict_upper_bound": value},
                        "require_strict_upper_bound",
                        True,
                    )


if __name__ == "__main__":
    unittest.main()
