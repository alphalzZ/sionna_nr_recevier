import unittest

import torch

from nr_pusch.noise import add_awgn, add_correlated_awgn, beam_noise_covariance


class AwgnTest(unittest.TestCase):
    def test_awgn_is_reproducible_and_uses_requested_snr(self):
        generator = torch.Generator().manual_seed(12)
        iq = torch.complex(
            torch.randn((2, 4, 120_000), generator=generator),
            torch.randn((2, 4, 120_000), generator=generator),
        )
        first = add_awgn(iq, 12.0, seed=4)
        second = add_awgn(iq, 12.0, seed=4)

        torch.testing.assert_close(first.iq, second.iq)
        torch.testing.assert_close(first.noise_variance, second.noise_variance)
        noise_power = (first.iq - iq).abs().square().mean(dim=-1)
        signal_power = iq.abs().square().mean(dim=-1)
        measured_snr = 10 * torch.log10(signal_power / noise_power)
        self.assertLess(float(torch.max(torch.abs(measured_snr - 12.0))), 0.15)
        self.assertEqual(tuple(first.noise_variance.shape), (2, 1, 4))



class CorrelatedBeamNoiseTest(unittest.TestCase):
    def test_covariance_and_empirical_shared_array_noise(self):
        generator = torch.Generator().manual_seed(17)
        weights = torch.complex(
            torch.randn((8, 4), generator=generator),
            torch.randn((8, 4), generator=generator),
        ) / (2.0 * 8.0**0.5)
        array_variance = 0.7
        post_variance = 0.01
        covariance = beam_noise_covariance(weights, array_variance, post_variance)
        torch.testing.assert_close(covariance, covariance.mH)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(covariance).min()), 0.0)

        noise = add_correlated_awgn(
            torch.zeros((200_000, 4), dtype=torch.complex64),
            weights,
            array_noise_variance=array_variance,
            post_noise_variance=post_variance,
            beam_axis=1,
            seed=13,
        )
        empirical = noise.mT @ noise.conj() / noise.shape[0]
        relative_error = torch.linalg.norm(empirical - covariance) / torch.linalg.norm(covariance)
        self.assertLessEqual(float(relative_error), 0.02)

    def test_noise_is_seeded_axis_aware_and_zero_variance_is_identity(self):
        weights = torch.eye(4, dtype=torch.complex64)
        signal = torch.ones((2, 5, 4, 3), dtype=torch.complex64)
        first = add_correlated_awgn(
            signal,
            weights,
            array_noise_variance=0.2,
            post_noise_variance=0.05,
            beam_axis=2,
            seed=29,
        )
        second = add_correlated_awgn(
            signal,
            weights,
            array_noise_variance=0.2,
            post_noise_variance=0.05,
            beam_axis=2,
            seed=29,
        )
        torch.testing.assert_close(first, second)
        self.assertEqual(first.shape, signal.shape)
        self.assertIs(
            add_correlated_awgn(
                signal,
                weights,
                array_noise_variance=0.0,
                post_noise_variance=0.0,
                beam_axis=2,
                seed=29,
            ),
            signal,
        )

    def test_rejects_invalid_covariance_and_noise_inputs(self):
        with self.assertRaisesRegex(ValueError, "有限非负数"):
            beam_noise_covariance(torch.eye(4, dtype=torch.complex64), -1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "长度必须为 4"):
            add_correlated_awgn(
                torch.zeros((2, 3), dtype=torch.complex64),
                torch.eye(4, dtype=torch.complex64),
                array_noise_variance=1.0,
                post_noise_variance=0.0,
                beam_axis=1,
                seed=1,
            )

if __name__ == "__main__":
    unittest.main()
