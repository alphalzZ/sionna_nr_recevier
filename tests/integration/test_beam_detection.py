from dataclasses import replace
from pathlib import Path
import unittest

import torch

from nr_pusch.config import TxSettings
from nr_pusch.receiver import NrPuschRx
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class BeamDetectorIntegrationTest(unittest.TestCase):
    def test_zf_and_beam_independent_decode_identity_channel(self):
        profiles = (
            ("dft_s_ofdm", "pusch_4ue.toml"),
            ("cp_ofdm", "pusch_rt_4ue_cp.toml"),
        )
        for waveform, profile in profiles:
            base = TxSettings.from_toml(ROOT / "configs" / profile)
            settings = replace(
                base,
                pusch=replace(
                    base.pusch,
                    waveform=waveform,
                    n_size_bwp=12,
                    mcs_index=8,
                ),
            )
            tx = NrPuschTx(settings, device="cpu")
            sent = tx.generate(batch_size=2, seed=13)
            num_users = len(settings.users)
            identity = torch.eye(num_users, dtype=sent.frequency_grid.dtype)
            received = torch.einsum(
                "ru,busf->brsf", identity, sent.frequency_grid[:, :, 0]
            ).unsqueeze(1)
            perfect_csi = identity.reshape(1, 1, num_users, num_users, 1, 1, 1).expand(
                2,
                1,
                num_users,
                num_users,
                1,
                sent.frequency_grid.shape[-2],
                sent.frequency_grid.shape[-1],
            )

            for estimator in ("perfect", "dmrs"):
                for detector in ("zf", "beam-independent"):
                    with self.subTest(
                        waveform=waveform, estimator=estimator, detector=detector
                    ):
                        receiver = NrPuschRx(
                            settings,
                            channel_estimator=estimator,
                            detector=detector,
                            input_domain="frequency",
                            device="cpu",
                        )
                        result = receiver.receive_frequency_grid(
                            received,
                            1e-8,
                            channel_frequency_response=(
                                perfect_csi if estimator == "perfect" else None
                            ),
                        )
                        self.assertTrue(torch.all(result.crc_status).item())
                        torch.testing.assert_close(result.bits, sent.bits, rtol=0, atol=0)


    def test_beam_independent_rejects_layer_and_observation_mismatch(self):
        multilayer = TxSettings.from_toml(
            ROOT / "configs" / "pusch_cp_2ue_2layer.toml"
        )
        with self.assertRaisesRegex(ValueError, "beam-independent 要求每用户单层"):
            NrPuschRx(multilayer, detector="beam-independent", device="cpu")

        base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        receiver = NrPuschRx(
            base,
            channel_estimator="perfect",
            detector="beam-independent",
            input_domain="frequency",
            device="cpu",
        )
        wrong_axis_grid = torch.zeros((1, 1, 3, 14, 600), dtype=torch.complex64)
        with self.assertRaisesRegex(ValueError, "beam-independent 要求每用户单层"):
            receiver.receive_frequency_grid(wrong_axis_grid, 1e-8)

if __name__ == "__main__":
    unittest.main()
