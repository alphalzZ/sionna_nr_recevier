import unittest

import torch

from nr_pusch.noise import add_awgn


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


if __name__ == "__main__":
    unittest.main()
