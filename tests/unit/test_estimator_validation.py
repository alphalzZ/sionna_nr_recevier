from __future__ import annotations

import unittest

import numpy as np
import torch

from nr_pusch.estimator_validation import (
    _acceptance_gate,
    _data_re_nmse,
    _paired_bootstrap_upper,
    _positive_override,
)


class EstimatorValidationGateTest(unittest.TestCase):
    def test_data_re_nmse_returns_per_frame_user_and_antenna_values(self):
        truth = torch.ones((2, 1, 2, 4, 1, 3, 5), dtype=torch.complex64)
        estimated = truth.clone()
        estimated[1, 0, 1, 2, 0] *= 2
        data_masks = torch.ones((4, 3, 5), dtype=torch.bool)

        nmse = _data_re_nmse(estimated, truth, data_masks)

        self.assertEqual(nmse.shape, (2, 4, 2))
        np.testing.assert_array_equal(nmse[0], np.zeros((4, 2)))
        self.assertEqual(nmse[1, 2, 1], 1.0)

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


if __name__ == "__main__":
    unittest.main()
