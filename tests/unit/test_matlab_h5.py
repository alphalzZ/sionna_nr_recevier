from pathlib import Path
import unittest

import numpy as np

from nr_pusch.iq import read_matlab_rx_reference, read_matlab_tx_reference


FIXTURE = Path(__file__).parents[1] / "fixtures" / "matlab_h5" / "TxTestVector.h5"
RX_FIXTURE = Path(__file__).parents[1] / "fixtures" / "matlab_h5" / "RxTestVector.h5"


class MatlabH5FixtureTest(unittest.TestCase):
    def test_matlab_rx_fixture_is_read_with_documented_axes_and_analysis_splits(self):
        reference = read_matlab_rx_reference(RX_FIXTURE)

        self.assertEqual(reference.frequency_grid.shape, (4, 14, 600))
        self.assertEqual(reference.data_grid.shape, (4, 13, 600))
        self.assertEqual(reference.pilot_grid.shape, (4, 600))
        self.assertTrue(np.iscomplexobj(reference.frequency_grid))
        np.testing.assert_array_equal(reference.frequency_grid[:, 2, :], reference.pilot_grid)
        np.testing.assert_array_equal(
            np.delete(reference.frequency_grid, 2, axis=1), reference.data_grid
        )

    def test_matlab_tx_fixture_is_read_with_documented_axes(self):
        reference = read_matlab_tx_reference(FIXTURE)

        self.assertEqual(reference.users, ("ue0", "ue1", "ue2", "ue3"))
        self.assertEqual(reference.frequency_grid.shape, (4, 14, 1, 600))
        self.assertEqual(reference.bits.shape, (4, 28168))
        self.assertTrue(np.iscomplexobj(reference.frequency_grid))
        self.assertTrue(set(np.unique(reference.bits)).issubset({0, 1}))

    def test_matlab_dmrs_symbol_has_expected_empty_re_elements(self):
        reference = read_matlab_tx_reference(FIXTURE)

        zeros_per_symbol = np.count_nonzero(np.abs(reference.frequency_grid) < 1e-12, axis=(2, 3))
        np.testing.assert_array_equal(zeros_per_symbol[:, 2], np.full(4, 300))
        self.assertEqual(np.count_nonzero(zeros_per_symbol), 4)


if __name__ == "__main__":
    unittest.main()
