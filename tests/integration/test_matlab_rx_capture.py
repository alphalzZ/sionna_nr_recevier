"""End-to-end decode of the supplied MATLAB receive capture."""

from pathlib import Path
import unittest

import torch

from nr_pusch.config import TxSettings
from nr_pusch.iq import read_matlab_rx_reference
from nr_pusch.receiver import NrPuschRx


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs" / "rx_pusch_4ue.toml"
FIXTURE = ROOT / "tests" / "fixtures" / "matlab_h5" / "RxTestVector.h5"


class MatlabRxCaptureTest(unittest.TestCase):
    def test_supplied_h5_capture_decodes_all_four_transport_blocks(self):
        settings = TxSettings.from_toml(CONFIG)
        reference = read_matlab_rx_reference(FIXTURE)
        received = torch.from_numpy(reference.frequency_grid[None, None, ...])
        receiver = NrPuschRx(
            settings,
            channel_estimator="dmrs",
            input_domain="frequency",
            device="cpu",
        )

        # The H5 vector has no noise-variance metadata and is noiseless.
        result = receiver.receive_frequency_grid(received, noise_variance=0.0)

        self.assertEqual(tuple(result.bits.shape), (1, 4, 28168))
        self.assertTrue(torch.all(result.crc_status).item())


if __name__ == "__main__":
    unittest.main()
