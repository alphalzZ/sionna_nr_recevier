import unittest

import numpy as np

from nr_pusch.rt_channel import _cp_energy_diagnostics


class CpEnergyDiagnosticsTest(unittest.TestCase):
    def test_cp_uses_tap_span_and_records_lag_bounds(self):
        taps = np.asarray([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.complex128)

        spans, bounds, outside, sufficient = _cp_energy_diagnostics(
            taps,
            l_min=-1,
            cyclic_prefix_length=2,
        )

        self.assertEqual(spans, [[3]])
        self.assertEqual(bounds, [[[-1, 2]]])
        self.assertAlmostEqual(outside[0][0], 0.5)
        self.assertFalse(sufficient)

    def test_cp_accepts_energy_interval_equal_to_prefix_span(self):
        taps = np.asarray([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.complex128)

        spans, bounds, outside, sufficient = _cp_energy_diagnostics(
            taps,
            l_min=-1,
            cyclic_prefix_length=3,
        )

        self.assertEqual(spans, [[3]])
        self.assertEqual(bounds, [[[-1, 2]]])
        self.assertEqual(outside, [[0.0]])
        self.assertTrue(sufficient)

    def test_zero_energy_channel_has_no_false_cp_failure(self):
        taps = np.zeros((1, 1, 4), dtype=np.complex128)

        spans, bounds, outside, sufficient = _cp_energy_diagnostics(
            taps,
            l_min=-1,
            cyclic_prefix_length=0,
        )

        self.assertEqual(spans, [[None]])
        self.assertEqual(bounds, [[None]])
        self.assertEqual(outside, [[0.0]])
        self.assertTrue(sufficient)


if __name__ == "__main__":
    unittest.main()
