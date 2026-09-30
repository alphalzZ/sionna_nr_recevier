from pathlib import Path
import unittest

import torch

from nr_pusch.config import TxSettings
from nr_pusch.iq import read_matlab_scrambling_sequences
from nr_pusch.receiver import NrPuschRx, _use_explicit_tx_scrambling
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class SoftMmsePicFeedbackTest(unittest.TestCase):
    def _profile_feedback(self, profile_name: str, *, explicit_scrambling: bool = False):
        settings = TxSettings.from_toml(ROOT / "configs" / profile_name)
        transmitter = NrPuschTx(settings, device="cpu")
        sequences = None
        if explicit_scrambling:
            sequences = torch.from_numpy(
                read_matlab_scrambling_sequences(
                    ROOT / "tests" / "fixtures" / "matlab_h5" / "scrambSeqCase123427.h5"
                )
            )
            _use_explicit_tx_scrambling(
                transmitter,
                sequences,
                len(settings.users),
                int(transmitter._tx_freq._tb_encoder.n),
            )
        receiver = NrPuschRx(
            settings,
            detector="soft-mmse-pic",
            num_decoder_iterations=8,
            device="cpu",
            scrambling_sequences=sequences,
        )
        return transmitter, receiver._receiver._mimo_detector._feedback

    def _check_codeword_feedback(self, profile_name: str, *, explicit_scrambling: bool = False):
        transmitter, feedback = self._profile_feedback(
            profile_name, explicit_scrambling=explicit_scrambling
        )
        encoder = transmitter._tx_freq._tb_encoder
        information = torch.randint(
            0,
            2,
            (2, 4, int(encoder.k)),
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(1027),
        )
        codeword = encoder(information)
        self.assertGreater(feedback._num_cbs, 0)
        self.assertGreaterEqual(feedback._num_fillers, 0)

        # Strong but imperfect observations exercise deinterleaving and BP
        # correction for independently scrambled users.
        channel_llr = (2.0 * codeword - 1.0) * 8.0
        flipped = torch.tensor([10, 1234, 5600, 19234], device=channel_llr.device)
        channel_llr[0, 0, flipped] *= -1.0
        extrinsic = feedback(channel_llr)
        self.assertEqual(tuple(extrinsic.shape), tuple(channel_llr.shape))
        self.assertTrue(torch.isfinite(extrinsic).all().item())
        self.assertTrue(
            torch.all((extrinsic[0, 0, flipped] > 0) == (codeword[0, 0, flipped] > 0)).item()
        )
        return encoder, feedback, codeword

    def test_rnti_scrambling_and_ldpc_extrinsic_correction(self):
        encoder, feedback, codeword = self._check_codeword_feedback(
            "rx_pusch_4ue.toml"
        )
        low_confidence = feedback((2.0 * codeword - 1.0) * 4.0)
        high_confidence = feedback((2.0 * codeword - 1.0) * 8.0)
        self.assertTrue(torch.all(low_confidence.abs() > high_confidence.abs()).item())
        torch.testing.assert_close(
            low_confidence.abs() - high_confidence.abs(),
            torch.full_like(low_confidence, 4.0),
            rtol=0,
            atol=0,
        )
        self.assertEqual(int(encoder.num_cbs), feedback._num_cbs)

    def test_explicit_scrambling_and_long_codeblock_layout(self):
        transmitter, feedback = self._profile_feedback(
            "rx_pusch_4ue_mcs27.toml", explicit_scrambling=True
        )
        encoder = transmitter._tx_freq._tb_encoder
        information = torch.randint(
            0,
            2,
            (1, 4, int(encoder.k)),
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(1027),
        )
        codeword = encoder(information)
        channel_llr = (2.0 * codeword - 1.0) * 8.0
        flipped = torch.tensor([19, 980, 11403, 30117], device=channel_llr.device)
        channel_llr[0, 0, flipped] *= -1.0
        extrinsic = feedback(channel_llr)
        self.assertEqual(feedback._num_fillers, 0)
        self.assertTrue(torch.isfinite(extrinsic).all().item())
        self.assertTrue(
            torch.all((extrinsic[0, 0, flipped] > 0) == (codeword[0, 0, flipped] > 0)).item()
        )


if __name__ == "__main__":
    unittest.main()
