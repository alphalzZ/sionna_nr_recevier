"""Unit tests for the BLER sweep stop policy, progress records, and export."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from nr_pusch import bler as bler_module
from nr_pusch.bler import BlerSweep, SkippedPoint, save_bler_results
from nr_pusch.cli import bler as bler_cli
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.simulation_config import BlerSettings


ROOT = Path(__file__).parents[2]
USERS = 4
BITS_PER_USER = 4


def _settings(**overrides) -> BlerSettings:
    values = dict(
        snr_db=(10.0, 20.0, 30.0),
        batch_size=2,
        max_frames_per_snr=2,
        target_block_errors=100,
        seed=1234,
        num_decoder_iterations=1,
        channel_estimator="dmrs",
        detector="lmmse",
        detector_parameter=None,
        detector_parameters={},
        detector_damping=0.25,
        detectors=("lmmse",),
        device="cpu",
        channel_domain="frequency",
    )
    values.update(overrides)
    return BlerSettings(**values)


@contextmanager
def _settings_files():
    yield (
        TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml"),
        ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_4x4.toml"),
    )


@contextmanager
def _fake_link(
    zero_bler_points: set[tuple[str, float] | tuple[str, str, float]],
):
    """Script clean outcomes by detector or by estimator-detector pair."""

    state: dict[str, float | None] = {"snr_db": None}

    class FakeTransmitter:
        sample_rate_hz = 18_000_000

        def __init__(self, settings, device=None):
            self._tx_freq = SimpleNamespace(resource_grid=SimpleNamespace(fft_size=512))

        def generate(self, batch_size: int, seed: int):
            return SimpleNamespace(
                bits=torch.zeros(batch_size, USERS, BITS_PER_USER),
                frequency_grid=SimpleNamespace(batch_size=batch_size, seed=seed),
                sample_rate_hz=self.sample_rate_hz,
            )

    class FakeChannel:
        def __init__(self, settings, device=None):
            pass

        def apply_frequency(self, frequency_grid, resource_grid):
            return SimpleNamespace(grid=frequency_grid, channel_frequency_response=None)

    class FakeReceiver:
        sample_rate_hz = 18_000_000

        def __init__(self, settings, **kwargs):
            self.channel_estimator = kwargs["channel_estimator"]
            self.detector = kwargs["detector"]

        def receive_frequency_grid(self, grid, noise_variance, channel_frequency_response=None):
            clean = (
                (self.channel_estimator, self.detector, state["snr_db"]) in zero_bler_points
                or (self.detector, state["snr_db"]) in zero_bler_points
            )
            crc_status = torch.ones(grid.batch_size, USERS, dtype=torch.bool)
            bits = torch.zeros(grid.batch_size, USERS, BITS_PER_USER)
            if not clean:
                crc_status[:] = False
                bits[:] = 1.0
            return SimpleNamespace(crc_status=crc_status, bits=bits)

    def fake_awgn(grid, snr_db, seed=0):
        state["snr_db"] = float(snr_db)
        return SimpleNamespace(grid=grid, noise_variance=0.01)

    with mock.patch.multiple(
        bler_module,
        NrPuschTx=FakeTransmitter,
        NrPuschCdlChannel=FakeChannel,
        NrPuschRx=FakeReceiver,
        add_awgn_resource_grid=fake_awgn,
    ):
        yield


class StopAtZeroBlerTest(unittest.TestCase):
    def test_zero_bler_point_skips_remaining_higher_snrs(self):
        settings = _settings(snr_db=(10.0, 20.0, 30.0, 40.0), stop_at_zero_bler=True)
        reported: list[tuple[float, bool]] = []
        with _settings_files() as (tx_settings, channel_settings), _fake_link({("lmmse", 20.0)}):
            sweep = bler_module.simulate_bler(
                tx_settings,
                channel_settings,
                settings,
                on_point=lambda point: reported.append((point.snr_db, point.block_errors == 0)),
                on_skip=lambda point: reported.append((point.snr_db, True)),
            )
        self.assertEqual([point.snr_db for point in sweep.points], [10.0, 20.0])
        self.assertEqual([point.block_errors for point in sweep.points], [8, 0])
        self.assertEqual(
            [(point.snr_db, point.trigger_snr_db, point.reason) for point in sweep.skipped],
            [(30.0, 20.0, "bler_at_zero"), (40.0, 20.0, "bler_at_zero")],
        )
        self.assertTrue(all(point.detector == "lmmse" and point.device == "cpu" for point in sweep.skipped))
        self.assertEqual(reported, [(10.0, False), (20.0, True), (30.0, True), (40.0, True)])

    def test_every_point_is_simulated_when_stop_is_disabled(self):
        settings = _settings(snr_db=(10.0, 20.0, 30.0), stop_at_zero_bler=False)
        skipped_reports: list[SkippedPoint] = []
        zero_bler = {("lmmse", 20.0), ("lmmse", 30.0)}
        with _settings_files() as (tx_settings, channel_settings), _fake_link(zero_bler):
            sweep = bler_module.simulate_bler(
                tx_settings, channel_settings, settings, on_skip=skipped_reports.append
            )
        self.assertEqual([point.snr_db for point in sweep.points], [10.0, 20.0, 30.0])
        self.assertEqual(sweep.skipped, ())
        self.assertEqual(skipped_reports, [])

    def test_zero_bler_at_highest_snr_skips_nothing(self):
        settings = _settings(snr_db=(10.0, 20.0), stop_at_zero_bler=True)
        with _settings_files() as (tx_settings, channel_settings), _fake_link({("lmmse", 20.0)}):
            sweep = bler_module.simulate_bler(tx_settings, channel_settings, settings)
        self.assertEqual([point.snr_db for point in sweep.points], [10.0, 20.0])
        self.assertEqual(sweep.skipped, ())

    def test_each_detector_applies_the_stop_policy_on_its_own_sweep(self):
        settings = _settings(
            snr_db=(10.0, 20.0, 30.0),
            detectors=("lmmse", "lmmse-sic"),
            stop_at_zero_bler=True,
        )
        zero_bler = {("lmmse", 20.0), ("lmmse-sic", 30.0)}
        with _settings_files() as (tx_settings, channel_settings), _fake_link(zero_bler):
            sweep = bler_module.simulate_detector_comparison(
                tx_settings, channel_settings, settings
            )
        self.assertEqual(
            [(point.detector, point.snr_db) for point in sweep.points],
            [
                ("lmmse", 10.0),
                ("lmmse", 20.0),
                ("lmmse-sic", 10.0),
                ("lmmse-sic", 20.0),
                ("lmmse-sic", 30.0),
            ],
        )
        self.assertEqual(
            [(point.detector, point.snr_db, point.trigger_snr_db) for point in sweep.skipped],
            [("lmmse", 30.0, 20.0)],
        )

    def test_estimator_detector_matrix_sweeps_every_pair_independently(self):
        settings = _settings(
            snr_db=(10.0, 20.0),
            channel_estimators=("dmrs", "dmrs-lmmse", "perfect"),
            dmrs_tap_power_prior_path="test-prior.npz",
            detectors=("lmmse", "soft-mmse-pic"),
            stop_at_zero_bler=True,
        )
        clean_point = ("perfect", "soft-mmse-pic", 10.0)
        with _settings_files() as (tx_settings, channel_settings):
            with (
                _fake_link({clean_point}),
                mock.patch.object(bler_module, "dmrs_prior_compatibility", return_value={}),
                mock.patch.object(
                    bler_module, "load_dmrs_tap_power_prior", return_value=(torch.ones(1), {})
                ),
            ):
                sweep = bler_module.simulate_detector_comparison(
                    tx_settings, channel_settings, settings
                )
        expected = [
            (estimator, detector, snr_db)
            for estimator in settings.channel_estimators_for_sweep
            for detector in settings.detectors
            for snr_db in settings.snr_db
            if (estimator, detector) != ("perfect", "soft-mmse-pic") or snr_db == 10.0
        ]
        self.assertEqual(
            [(point.channel_estimator, point.detector, point.snr_db) for point in sweep.points],
            expected,
        )
        self.assertEqual(
            [
                (point.channel_estimator, point.detector, point.snr_db, point.trigger_snr_db)
                for point in sweep.skipped
            ],
            [("perfect", "soft-mmse-pic", 20.0, 10.0)],
        )
        first_snr_errors = {
            (point.channel_estimator, point.detector): point.block_errors
            for point in sweep.points
            if point.snr_db == 10.0
        }
        self.assertEqual(first_snr_errors[("perfect", "soft-mmse-pic")], 0)
        self.assertEqual(first_snr_errors[("dmrs", "lmmse")], 8)



class BlerResultExportTest(unittest.TestCase):
    def _sweep(self) -> BlerSweep:
        return BlerSweep(
            points=(
                bler_module.BlerPoint(
                    detector="lmmse", device="cpu", snr_db=10.0, frames=2,
                    transport_blocks=8, block_errors=8, crc_failures=8,
                    bit_errors=32, bits=32, bler=1.0, crc_fail_rate=1.0,
                    ber=1.0, runtime_s=0.5, channel_estimator="dmrs-lmmse",
                ),
            ),
            skipped=(
                SkippedPoint(
                    detector="lmmse", device="cpu", snr_db=20.0,
                    trigger_snr_db=10.0, reason="bler_at_zero",
                    channel_estimator="dmrs-lmmse",
                ),
            ),
        )

    def test_csv_keeps_measurements_and_manifest_lists_skipped_points(self):
        settings = _settings()
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with _settings_files() as (tx_settings, channel_settings):
            csv_path, manifest_path = save_bler_results(
                self._sweep(), directory / "bler.csv",
                tx_settings=tx_settings, channel_settings=channel_settings,
                simulation_settings=settings,
            )
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["snr_db"] for row in rows], ["10.0"])
        self.assertEqual(rows[0]["channel_estimator"], "dmrs-lmmse")
        self.assertEqual(list(rows[0]), list(bler_module.BlerPoint.__dataclass_fields__))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual([entry["snr_db"] for entry in manifest["skipped_points"]], [20.0])
        self.assertEqual(manifest["skipped_points"][0]["trigger_snr_db"], 10.0)
        self.assertEqual(manifest["skipped_points"][0]["reason"], "bler_at_zero")
        self.assertIn("stop_at_zero_bler", manifest["skip_policy"])
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["results"][0]["channel_estimator"], "dmrs-lmmse")
        self.assertFalse(manifest["simulation_settings"]["stop_at_zero_bler"])

    def test_cp_manifest_uses_per_re_detector_validity_note(self):
        tx_settings = TxSettings.from_toml(
            ROOT / "configs" / "pusch_cp_2ue_2layer.toml"
        )
        channel_settings = ChannelSettings.from_toml(
            ROOT / "configs" / "cdl_38_901_2tx_4rx.toml"
        )
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _, manifest_path = save_bler_results(
            self._sweep(),
            directory / "cp-bler.csv",
            tx_settings=tx_settings,
            channel_settings=channel_settings,
            simulation_settings=_settings(),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        note = manifest["detector_validity_note"]
        self.assertIn("CP-OFDM", note)
        self.assertIn("per-resource-element", note)
        self.assertIn("double precision", note)
        self.assertNotIn("DFT-s-OFDM", note)
        self.assertNotIn("IDFT", note)


class BlerCliProgressTest(unittest.TestCase):
    def test_progress_jsonl_marks_measured_and_skipped_points(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        simulation = directory / "simulation.toml"
        simulation.write_text(
            "\n".join(
                [
                    "[bler]",
                    "snr_db = [10.0, 20.0, 30.0]",
                    "batch_size = 2",
                    "max_frames_per_snr = 2",
                    "target_block_errors = 100",
                    "seed = 1234",
                    "num_decoder_iterations = 1",
                    'channel_estimator = "dmrs"',
                    'detector = "lmmse"',
                    "stop_at_zero_bler = true",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        progress = directory / "progress.jsonl"
        argv = [
            "nr-pusch-bler",
            "--tx-config", str(ROOT / "configs" / "pusch_4ue.toml"),
            "--channel-config", str(ROOT / "configs" / "cdl_38_901_4x4.toml"),
            "--simulation-config", str(simulation),
            "--output", str(directory / "results.csv"),
            "--progress-jsonl", str(progress),
        ]
        with _fake_link({("lmmse", 20.0)}), mock.patch.object(sys, "argv", argv):
            bler_cli.main()
        records = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(
            [(record["snr_db"], record["skipped"]) for record in records],
            [(10.0, False), (20.0, False), (30.0, True)],
        )
        self.assertEqual(records[0]["channel_estimator"], "dmrs")
        self.assertEqual(records[2]["trigger_snr_db"], 20.0)
        self.assertNotIn("bler", records[2])
        with (directory / "results.csv").open(newline="", encoding="utf-8") as stream:
            self.assertEqual([row["snr_db"] for row in csv.DictReader(stream)], ["10.0", "20.0"])


if __name__ == "__main__":
    unittest.main()
