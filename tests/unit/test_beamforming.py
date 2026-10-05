import unittest

import numpy as np

from nr_pusch.beamforming import combine_beams, compute_beam_metrics, make_beam_weights


class BeamformingTest(unittest.TestCase):
    def setUp(self):
        rows, cols = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
        self.positions = np.column_stack(
            (np.zeros(64), (cols.ravel() - 3.5) * 0.5 * 299_792_458.0 / 3.5e9,
             (rows.ravel() - 3.5) * 0.5 * 299_792_458.0 / 3.5e9)
        )
        angles = np.deg2rad([-40.0, -15.0, 15.0, 40.0])
        self.directions = np.column_stack(
            (np.cos(angles), np.sin(angles), np.zeros(4))
        )

    def test_shared_combining_matches_matrix_algebra(self):
        rng = np.random.default_rng(13)
        h_ant = rng.normal(size=(64, 4, 12)) + 1j * rng.normal(size=(64, 4, 12))
        symbols = rng.normal(size=(4, 12)) + 1j * rng.normal(size=(4, 12))
        weights = make_beam_weights(self.positions, self.directions, 3.5e9)
        h_beam = combine_beams(h_ant, weights)
        h_applied = np.einsum("njt,jt->nt", h_ant, symbols)
        lhs = combine_beams(h_applied[:, None, :], weights)[:, 0, :]
        rhs = np.einsum("bjt,jt->bt", h_beam, symbols)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)

    def test_beam_gram_matrix_is_hermitian_positive_semidefinite(self):
        weights = make_beam_weights(self.positions, self.directions, 3.5e9)
        gram = weights.conj().T @ weights
        np.testing.assert_allclose(gram, gram.conj().T, rtol=0.0, atol=1e-13)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(gram).min()), -1e-12)

    def test_matched_plane_wave_has_unit_response_and_energy_identity(self):
        weights = make_beam_weights(self.positions, self.directions, 3.5e9)
        wavelength = 299_792_458.0 / 3.5e9
        direction = self.directions[0]
        array_response = np.exp(1j * 2.0 * np.pi * (self.positions @ direction) / wavelength)
        np.testing.assert_allclose(weights[:, 0].conj() @ array_response / np.sqrt(64), 1.0, atol=1e-12)

        vector = np.arange(64) + 1j * np.arange(64)[::-1]
        left = np.linalg.norm(weights.conj().T @ vector) ** 2
        right = np.vdot(vector, weights @ weights.conj().T @ vector).real
        self.assertAlmostEqual(float(left), float(right), places=9)

    def test_one_silent_user_contributes_zero_to_every_beam(self):
        weights = make_beam_weights(self.positions, self.directions, 3.5e9)
        h_ant = np.ones((64, 4, 3), dtype=np.complex128)
        h_ant[:, 2, :] = 0.0
        h_beam = combine_beams(h_ant, weights)
        np.testing.assert_array_equal(h_beam[:, 2, :], np.zeros((4, 3), dtype=np.complex128))

    def test_phase_quantization_preserves_constant_modulus(self):
        weights = make_beam_weights(self.positions, self.directions, 3.5e9, phase_bits=3)
        np.testing.assert_allclose(np.abs(weights), 1.0 / 8.0, rtol=0.0, atol=1e-14)

    def test_rejects_malformed_or_nonfinite_inputs(self):
        with self.assertRaisesRegex(ValueError, "element_positions_m"):
            make_beam_weights(self.positions[:3], self.directions, 3.5e9)
        bad_directions = self.directions.copy()
        bad_directions[0] = 0.0
        with self.assertRaisesRegex(ValueError, "不能为零"):
            make_beam_weights(self.positions, bad_directions, 3.5e9)
        with self.assertRaisesRegex(ValueError, "复数数组"):
            combine_beams(np.ones((64, 4)), np.ones((64, 4)))


    def test_beam_metrics_follow_unwhitened_power_formulas(self):
        h_beam = np.zeros((4, 4, 2), dtype=np.complex128)
        for user in range(4):
            h_beam[user, user, :] = user + 2
        h_beam[0, 1, :] = 1.0
        h_beam[1, 0, :] = 0.5
        metrics = compute_beam_metrics(h_beam, np.array([0.0, 10.0, 20.0, 30.0]))
        self.assertAlmostEqual(metrics["leakage_ratio_l_by_beam_user"][0][1], 1.0 / 9.0)
        self.assertAlmostEqual(
            metrics["interference_to_signal_ratio_by_beam_user"][0][1],
            2.5,
        )
        self.assertAlmostEqual(metrics["interference_power_by_beam"][0], 10.0)
        self.assertEqual(metrics["minimum_rank"], 4)

    def test_beam_metrics_preserve_nulls_without_epsilon(self):
        h_beam = np.eye(4, dtype=np.complex128)
        h_beam[0, 0] = 0.0
        metrics = compute_beam_metrics(h_beam, np.zeros(4))
        self.assertIsNone(metrics["leakage_ratio_l_by_beam_user"][0][0])
        self.assertIsNone(metrics["interference_to_signal_ratio_by_beam_user"][0][2])

if __name__ == "__main__":
    unittest.main()
