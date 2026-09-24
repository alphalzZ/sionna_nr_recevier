import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from nr_pusch.artifacts import save_tx_result
from nr_pusch.config import TxSettings
from nr_pusch.transmitter import TxResult


ROOT = Path(__file__).parents[2]


class ArtifactExportTest(unittest.TestCase):
    def test_npz_and_manifest_preserve_signal_arrays_and_run_config(self):
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        result = TxResult(
            iq=torch.zeros((1, 4, 1, 8), dtype=torch.complex64),
            frequency_grid=torch.zeros((1, 4, 1, 14, 600), dtype=torch.complex64),
            bits=torch.zeros((1, 4, 2), dtype=torch.float32),
            sample_rate_hz=18_000_000,
            metadata={"waveform": "dft_s_ofdm"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            archive_path, manifest_path = save_tx_result(result, settings, Path(tmp) / "tx.npz")
            with np.load(archive_path) as archive:
                self.assertEqual(archive["iq"].shape, (1, 4, 1, 8))
                self.assertEqual(archive["frequency_grid"].shape, (1, 4, 1, 14, 600))
                self.assertEqual(archive["bits"].shape, (1, 4, 2))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["settings"]["pusch"]["waveform"], "dft_s_ofdm")
            self.assertEqual(manifest["result"]["waveform"], "dft_s_ofdm")
            self.assertEqual(manifest["artifacts"]["arrays"]["frequency_grid"], "frequency_grid")


if __name__ == "__main__":
    unittest.main()
