import dataclasses
import unittest
from pathlib import Path

import torch
from sionna.phy.mapping import Demapper

from nr_pusch.receiver import beam_independent_equalizer
from nr_pusch.simulation_config import BlerSettings


ROOT = Path(__file__).parents[2]


class BeamIndependentEqualizerTest(unittest.TestCase):
    def test_off_diagonal_stream_power_is_in_effective_noise(self):
        h = torch.diag(torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.complex64))
        h[0, 1] = 1.0 + 0.0j
        h[0, 2] = 0.0 + 0.5j
        x = torch.tensor([1.0 + 1.0j, -1.0 + 0.5j, 0.5 - 1.0j, 1.0 - 0.5j])
        y = h @ x
        s = torch.eye(4, dtype=torch.complex64) * 0.2

        x_hat, no_eff = beam_independent_equalizer(y, h, s)

        self.assertEqual(x_hat.shape, (4,))
        torch.testing.assert_close(x_hat[0], y[0] / h[0, 0])
        self.assertAlmostEqual(float(no_eff[0]), (1.0 + 0.25 + 0.2) / 4.0)
        torch.testing.assert_close(
            no_eff[1:],
            torch.tensor([0.2 / 9.0, 0.2 / 16.0, 0.2 / 25.0]),
        )

    def test_zero_desired_gain_is_an_erasure_with_finite_zero_llr(self):
        h = torch.eye(4, dtype=torch.complex64)
        h[2, 2] = 0.0
        h[2, 0] = 0.5
        y = torch.ones(4, dtype=torch.complex64)
        s = torch.eye(4, dtype=torch.complex64) * 0.1

        x_hat, no_eff = beam_independent_equalizer(y, h, s)
        llr = Demapper("app", "qam", 2, device="cpu")(
            x_hat.reshape(1, 1, 1, 4), no_eff.reshape(1, 1, 1, 4)
        )

        self.assertEqual(complex(x_hat[2]), 0j)
        self.assertTrue(torch.isinf(no_eff[2]).item())
        self.assertTrue(torch.isfinite(llr).all().item())
        torch.testing.assert_close(llr[..., 4:6], torch.zeros_like(llr[..., 4:6]))

    def test_rejects_unequal_observation_and_stream_axes(self):
        y = torch.ones(4, dtype=torch.complex64)
        h = torch.ones((4, 3), dtype=torch.complex64)
        s = torch.eye(4, dtype=torch.complex64)
        with self.assertRaisesRegex(ValueError, "beam-independent 要求每用户单层"):
            beam_independent_equalizer(y, h, s)


class BeamDetectorConfigTest(unittest.TestCase):
    def test_new_detectors_accept_only_parameterless_configuration(self):
        base = BlerSettings.from_toml(ROOT / "configs" / "bler_smoke.toml")
        for detector in ("beam-independent", "zf"):
            with self.subTest(detector=detector):
                settings = dataclasses.replace(
                    base,
                    detector=detector,
                    detectors=(detector,),
                    detector_parameter=None,
                    detector_parameters={},
                )
                settings.validate()
                with self.assertRaisesRegex(ValueError, "detector_parameter"):
                    dataclasses.replace(
                        settings, detector_parameter=2
                    ).validate()
                with self.assertRaisesRegex(ValueError, "detector_parameters"):
                    dataclasses.replace(
                        settings, detector_parameters={detector: 2}
                    ).validate()


if __name__ == "__main__":
    unittest.main()
