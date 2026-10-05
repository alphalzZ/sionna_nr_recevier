from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nr_pusch.beam_simulation import (
    make_rt_beam_manifest_metadata,
    save_rt_beam_bler,
    save_rt_beam_diagnostic_captures,
    simulate_rt_beam_bler,
)
from nr_pusch.config import TxSettings
from nr_pusch.rt_channel import RtBeamSnapshot
from nr_pusch.simulation_config import BlerSettings
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).resolve().parents[2]


def identity_snapshot(tx_settings: TxSettings) -> RtBeamSnapshot:
    transmitter = NrPuschTx(tx_settings, device="cpu")
    resource_grid = transmitter._tx_freq.resource_grid
    fft_size = int(resource_grid.fft_size)
    num_symbols = int(resource_grid.num_ofdm_symbols)
    sample_rate_hz = int(fft_size * resource_grid.subcarrier_spacing)
    h_beam = np.broadcast_to(np.eye(4, dtype=np.complex128)[:, :, None], (4, 4, fft_size)).copy()
    zero_paths = np.zeros((4, 4, 1), dtype=np.complex128)
    users = [
        {"name": user.name, "power_scale_db": 0.0}
        for user in tx_settings.users
    ]
    metadata = {
        "scene": "empty",
        "users": [user.name for user in tx_settings.users],
        "fft_size": fft_size,
        "num_ofdm_symbols": num_symbols,
        "sample_rate_hz": sample_rate_hz,
        "carrier_frequency_hz": 3.5e9,
        "l_min": -6,
        "l_max": 20,
        "max_delay_spread_s": 3e-6,
        "cp_sufficient": True,
        "config_sha256": "controlled-identity-channel",
        "rt_config": {
            "receiver": {"l_min": -6, "max_delay_spread_s": 3e-6},
            "users": users,
            "noise": {"post_combiner_ratio": 0.001},
        },
    }
    return RtBeamSnapshot(
        path_a=zero_paths.copy(),
        path_tau_s=np.zeros((4, 4, 1), dtype=np.float64),
        path_valid=np.ones((4, 4, 1), dtype=np.bool_),
        ta_s=np.zeros(4, dtype=np.float64),
        element_positions_m=np.zeros((4, 3), dtype=np.float64),
        weights=np.eye(4, dtype=np.complex128),
        h_ant=np.zeros((4, 4, fft_size), dtype=np.complex128),
        h_beam=h_beam,
        taps_ant=np.zeros((4, 4, 1), dtype=np.complex128),
        taps_beam=np.zeros((4, 4, 1), dtype=np.complex128),
        frequencies_hz=np.zeros(fft_size, dtype=np.float64),
        path_theta_r_rad=np.zeros((4, 4, 1), dtype=np.float64),
        path_phi_r_rad=np.zeros((4, 4, 1), dtype=np.float64),
        path_vertices_m=np.zeros((4, 4, 1, 1, 3), dtype=np.float64),
        path_interactions=np.zeros((4, 4, 1), dtype=np.int64),
        metadata=metadata,
    )


class RtBeamReceiverIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulation_settings = BlerSettings.from_toml(ROOT / "configs/bler_rt_beam_smoke.toml")

    def test_identity_channel_decodes_dft_and_cp_pusch_for_all_three_beam_arms(self):
        settings = replace(
            self.simulation_settings,
            snr_db=(80.0,),
            batch_size=1,
            max_frames_per_snr=2,
            target_block_errors=8,
            channel_estimators=("perfect", "dmrs"),
            detectors=("beam-independent", "zf", "lmmse"),
        )
        settings.validate()
        for tx_name in ("pusch_4ue.toml", "pusch_rt_4ue_cp.toml"):
            with self.subTest(waveform=tx_name):
                tx_settings = TxSettings.from_toml(ROOT / "configs" / tx_name)
                snapshot = identity_snapshot(tx_settings)
                rows = simulate_rt_beam_bler(
                    tx_settings,
                    snapshot,
                    settings,
                    post_combiner_ratio=0.001,
                    seed=13,
                    device="cpu",
                )
                self.assertEqual(len(rows), 2 * 3 * 5)
                for index in range(0, len(rows), 5):
                    point = rows[index:index + 5]
                    self.assertEqual([row.user for row in point], [
                        user.name for user in tx_settings.users
                    ] + ["all"])
                    self.assertTrue(all(row.status == "complete" for row in point))
                    self.assertTrue(all(row.block_errors == 0 for row in point))
                    self.assertTrue(all(row.crc_failures == 0 for row in point))
                    self.assertTrue(all(row.frames == 2 for row in point))
                    self.assertEqual(point[-1].transport_blocks, 8)
                    self.assertEqual(point[-1].bler, sum(row.bler for row in point[:4]) / 4)
                    self.assertGreater(point[-1].bler_ci95_high, 0.0)

                metadata = make_rt_beam_manifest_metadata(
                    tx_settings,
                    snapshot,
                    settings,
                    post_combiner_ratio=0.001,
                    seed=13,
                    device="cpu",
                )
                with tempfile.TemporaryDirectory() as directory:
                    csv_path, manifest_path = save_rt_beam_bler(
                        rows, Path(directory) / "result.csv", metadata=metadata
                    )
                    self.assertTrue(csv_path.is_file())
                    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
                    self.assertEqual(saved["channel_sampling"], "fixed_rt_snapshot")
                    np.testing.assert_allclose(
                        np.asarray(saved["noise_covariance_by_snr_db"]["80"]["real"]),
                        np.eye(4) * 1e-8,
                        rtol=1e-12,
                        atol=1e-20,
                    )
                    self.assertEqual(saved["result"]["row_count"], len(rows))


    def test_stop_at_zero_marks_higher_snr_unmeasured(self):
        tx_settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        snapshot = identity_snapshot(tx_settings)
        settings = replace(
            self.simulation_settings,
            snr_db=(80.0, 90.0),
            batch_size=1,
            max_frames_per_snr=2,
            target_block_errors=8,
            channel_estimators=("perfect",),
            detectors=("lmmse",),
            stop_at_zero_bler=True,
            device="cpu",
        )
        rows = simulate_rt_beam_bler(
            tx_settings, snapshot, settings, post_combiner_ratio=0.001, seed=13, device="cpu"
        )
        measured, skipped = rows[:5], rows[5:]
        self.assertEqual([row.status for row in measured], ["complete"] * 5)
        self.assertEqual(measured[-1].block_errors, 0)
        self.assertEqual([row.status for row in skipped], ["skipped"] * 5)
        self.assertTrue(all(row.frames == 0 and row.transport_blocks == 0 for row in skipped))
        self.assertTrue(all(row.bler is None and row.bler_ci95_high is None for row in skipped))
        self.assertTrue(all(row.reason and "80 dB" in row.reason for row in skipped))

        metadata = make_rt_beam_manifest_metadata(
            tx_settings, snapshot, settings,
            post_combiner_ratio=0.001, seed=13, device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path, manifest_path = save_rt_beam_bler(
                rows, Path(directory) / "stopped.csv", metadata=metadata
            )
            with csv_path.open(newline="", encoding="utf-8") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(csv_rows[-1]["status"], "skipped")
            self.assertEqual(csv_rows[-1]["bler"], "")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["result"]["status_counts"]["skipped"], 5)
            self.assertEqual(
                manifest["result"]["skipped_points"],
                [{
                    "channel_estimator": "perfect",
                    "detector": "lmmse",
                    "snr_db": 90.0,
                    "reason": skipped[-1].reason,
                }],
            )
    def test_diagnostic_captures_preserve_payload_and_replay_coordinates(self):
        tx_settings = TxSettings.from_toml(ROOT / "configs/pusch_4ue.toml")
        snapshot = identity_snapshot(tx_settings)
        settings = replace(self.simulation_settings, snr_db=(80.0,))
        with tempfile.TemporaryDirectory() as directory:
            capture_paths = save_rt_beam_diagnostic_captures(
                tx_settings,
                snapshot,
                settings,
                directory,
                post_combiner_ratio=0.001,
                seed=13,
                device="cpu",
            )
            self.assertEqual(capture_paths["whitening_status"], "available")
            self.assertIsNotNone(capture_paths["whitened"])
            captures = {}
            for basis in ("physical_beam", "diagonal_noise_scaled", "whitened"):
                with np.load(capture_paths[basis], allow_pickle=False) as archive:
                    captures[basis] = {key: np.array(archive[key], copy=True) for key in archive.files}
                sidecar = json.loads(
                    Path(capture_paths[basis]).with_suffix(".json").read_text(encoding="utf-8")
                )
                self.assertEqual(sidecar["observation_basis"], basis)
                self.assertEqual(sidecar["input_domain"], "frequency")
                self.assertEqual(sidecar["noise_variance_for_replay"], (
                    None if basis == "physical_beam" else 1.0
                ))

        physical = captures["physical_beam"]
        diagonal = captures["diagonal_noise_scaled"]
        whitened = captures["whitened"]
        for capture in captures.values():
            self.assertEqual(capture["grid"].shape[1:3], (1, 4))
            self.assertEqual(capture["channel_frequency_response"].shape[1:5], (1, 4, 4, 1))
            np.testing.assert_array_equal(capture["bits"], physical["bits"])
        covariance = physical["noise_covariance"]
        diagonal_scale = np.sqrt(np.diag(covariance).real)
        expected_diagonal_grid = physical["grid"] / diagonal_scale[None, None, :, None, None]
        np.testing.assert_allclose(diagonal["grid"], expected_diagonal_grid, rtol=1e-5, atol=1e-7)
        cholesky = np.linalg.cholesky(covariance)
        grid_beam_first = physical["grid"].transpose(2, 0, 1, 3, 4)
        expected_white_grid = np.linalg.solve(
            cholesky, grid_beam_first.reshape(4, -1)
        ).reshape(grid_beam_first.shape).transpose(1, 2, 0, 3, 4)
        np.testing.assert_allclose(whitened["grid"], expected_white_grid, rtol=1e-5, atol=1e-7)
        csi_beam_first = physical["channel_frequency_response"].transpose(2, 0, 1, 3, 4, 5, 6)
        expected_white_csi = np.linalg.solve(
            cholesky, csi_beam_first.reshape(4, -1)
        ).reshape(csi_beam_first.shape).transpose(1, 2, 0, 3, 4, 5, 6)
        np.testing.assert_allclose(
            whitened["channel_frequency_response"], expected_white_csi,
            rtol=1e-5, atol=1e-7,
        )

    def test_rejects_cdl_prior_and_snapshot_delay_mismatch_before_decode(self):
        tx_settings = TxSettings.from_toml(ROOT / "configs/pusch_4ue.toml")
        snapshot = identity_snapshot(tx_settings)
        with self.assertRaisesRegex(ValueError, "RT beam channel 不支持 CDL dmrs-lmmse prior；请使用 dmrs 或 perfect"):
            simulate_rt_beam_bler(
                tx_settings,
                snapshot,
                replace(self.simulation_settings, channel_estimators=("dmrs-lmmse",)),
                post_combiner_ratio=0.001,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "max_delay_spread_s"):
            simulate_rt_beam_bler(
                tx_settings,
                snapshot,
                replace(self.simulation_settings, max_delay_spread_s=2e-6),
                post_combiner_ratio=0.001,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
