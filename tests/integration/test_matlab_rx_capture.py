"""End-to-end decode of the supplied MATLAB receive capture."""

from pathlib import Path
import tomllib
import unittest

import numpy as np
import torch

from nr_pusch.config import TxSettings
from nr_pusch.iq import read_matlab_rx_reference
from nr_pusch.receiver import NrPuschRx


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs" / "rx_pusch_4ue.toml"
FIXTURE = ROOT / "tests" / "fixtures" / "matlab_h5" / "RxTestVector.h5"


class MatlabRxCaptureTest(unittest.TestCase):
    def test_rx_profile_decodes_h5_reference_bits_and_crc(self):
        settings = TxSettings.from_toml(CONFIG)
        reference = read_matlab_rx_reference(FIXTURE)
        with CONFIG.open("rb") as stream:
            receiver_profile = tomllib.load(stream)["receiver"]
        self.assertIsNotNone(reference.transmitted_bits)
        received = torch.from_numpy(reference.frequency_grid[None, None, ...])
        receiver = NrPuschRx(
            settings,
            channel_estimator=receiver_profile["channel_estimator"],
            detector=receiver_profile["detector"],
            detector_parameter=receiver_profile["detector_parameter"],
            detector_damping=receiver_profile["detector_damping"],
            input_domain=receiver_profile["input_domain"],
            max_delay_spread_s=receiver_profile["max_delay_spread_s"],
            device="cpu",
        )
        result = receiver.receive_frequency_grid(
            received, noise_variance=receiver_profile["noise_variance"]
        )

        self.assertEqual(tuple(result.bits.shape), (1, 4, 28168))
        self.assertTrue(torch.isfinite(result.constellation.real).all().item())
        self.assertTrue(torch.isfinite(result.constellation.imag).all().item())
        self.assertTrue(torch.all(result.crc_status).item())
        np.testing.assert_array_equal(
            result.bits.detach().cpu().numpy()[0].astype(np.uint8),
            reference.transmitted_bits,
        )


if __name__ == "__main__":
    unittest.main()
