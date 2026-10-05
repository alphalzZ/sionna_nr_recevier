from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from nr_pusch.beam_validation import (
    _web_json_safe,
    validate_rt_web_snapshot,
)
from nr_pusch.beamforming import combine_beams
from nr_pusch.config import TxSettings
from nr_pusch.rt_channel import _cp_energy_diagnostics, prepare_rt_beam_snapshot
from nr_pusch.rt_config import RtBeamSettings


ROOT = Path(__file__).resolve().parents[2]


class RtWebValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tx_settings = TxSettings.from_toml(ROOT / "configs/pusch_4ue.toml")
        cls.rt_settings = RtBeamSettings.from_toml(ROOT / "configs/rt_beam_los.toml")
        cls.snapshot = prepare_rt_beam_snapshot(cls.tx_settings, cls.rt_settings)

    def _snapshot_with_taps(self, mutate):
        taps_beam = self.snapshot.taps_beam.copy()
        mutate(taps_beam)
        weights = np.asarray(self.snapshot.weights)
        gram = weights.conj().T @ weights
        delta = taps_beam - self.snapshot.taps_beam
        taps_ant = self.snapshot.taps_ant.copy()
        for tap_index in range(delta.shape[-1]):
            for user_index in range(delta.shape[1]):
                taps_ant[:, user_index, tap_index] += (
                    weights @ np.linalg.solve(gram, delta[:, user_index, tap_index])
                )
        lags = np.arange(
            int(self.snapshot.metadata["l_min"]),
            int(self.snapshot.metadata["l_max"]) + 1,
        )
        phase = np.exp(
            -2j
            * np.pi
            * self.snapshot.frequencies_hz[None, None, :, None]
            * lags[None, None, None, :]
            / int(self.snapshot.metadata["sample_rate_hz"])
        )
        h_ant = np.sum(taps_ant[:, :, None, :] * phase, axis=-1)
        h_beam = combine_beams(h_ant, weights)
        metadata = dict(self.snapshot.metadata)
        spans, bounds, outside, sufficient = _cp_energy_diagnostics(
            taps_beam,
            l_min=int(metadata["l_min"]),
            cyclic_prefix_length=int(metadata["cyclic_prefix_length_samples"]),
        )
        metadata.update(
            cp_energy_interval_samples_by_beam_user=spans,
            cp_energy_interval_bounds_samples_by_beam_user=bounds,
            cp_outside_energy_ratio_by_beam_user=outside,
            cp_outside_energy_ratio_max=max(max(row) for row in outside),
            cp_sufficient=sufficient,
        )
        return replace(
            self.snapshot,
            taps_ant=taps_ant,
            taps_beam=taps_beam,
            h_ant=h_ant,
            h_beam=h_beam,
            metadata=metadata,
        )


    def test_small_late_echo_passes_web_gate_but_not_strict_gate(self):
        echo = self._snapshot_with_taps(
            lambda taps: taps.__setitem__(
                (0, 0, -1), taps[0, 0, -1] + 0.002 * float(np.max(np.abs(taps)))
            )
        )
        report = validate_rt_web_snapshot(self.tx_settings, self.rt_settings, echo)
        rms = report["strict_time_report"]["fd_td_relative_rms_error"]
        self.assertTrue(report["passed"], report)
        self.assertGreater(rms, 1e-5)
        self.assertLessEqual(rms, 1e-3)
        self.assertFalse(report["strict_fd_td_passed"])
        self.assertTrue(report["warnings"])

    def test_web_gate_rejects_excessive_rms_cp_and_cfr_errors(self):
        echo = self._snapshot_with_taps(
            lambda taps: taps.__setitem__(
                (0, 0, -1), taps[0, 0, -1] + 0.02 * float(np.max(np.abs(taps)))
            )
        )
        rms_report = validate_rt_web_snapshot(self.tx_settings, self.rt_settings, echo)
        self.assertGreater(
            rms_report["strict_time_report"]["fd_td_relative_rms_error"], 1e-3
        )
        self.assertFalse(rms_report["passed"])

        cp_metadata = dict(self.snapshot.metadata)
        cp_metadata["cp_sufficient"] = False
        cp_report = validate_rt_web_snapshot(
            self.tx_settings, self.rt_settings, replace(self.snapshot, metadata=cp_metadata)
        )
        self.assertFalse(cp_report["passed"])

        cfr_snapshot = replace(
            self.snapshot,
            h_ant=self.snapshot.h_ant * 1.02,
            h_beam=self.snapshot.h_beam * 1.02,
        )
        cfr_report = validate_rt_web_snapshot(
            self.tx_settings, self.rt_settings, cfr_snapshot
        )
        self.assertFalse(cfr_report["passed"])
        cfr_check = next(
            check for check in cfr_report["checks"]
            if check["check"] == "direct_cfr_vs_finite_tap_cfr"
        )
        self.assertGreater(cfr_check["actual"], 0.01)

    def test_missing_and_nan_cp_diagnostics_fail_closed_and_serialize(self):
        missing_metadata = dict(self.snapshot.metadata)
        missing_metadata.pop("cp_outside_energy_ratio_max")
        missing_report = validate_rt_web_snapshot(
            self.tx_settings,
            self.rt_settings,
            replace(self.snapshot, metadata=missing_metadata),
        )
        self.assertFalse(missing_report["passed"])
        json.dumps(missing_report, allow_nan=False)

        nan_metadata = dict(self.snapshot.metadata)
        nan_metadata["cp_outside_energy_ratio_max"] = float("nan")
        nan_report = validate_rt_web_snapshot(
            self.tx_settings,
            self.rt_settings,
            replace(self.snapshot, metadata=nan_metadata),
        )
        self.assertFalse(nan_report["passed"])
        cp_check = next(
            check for check in nan_report["checks"]
            if check["check"] == "cyclic_prefix_outside_effective_energy"
        )
        self.assertIsNone(cp_check["actual"])
        json.dumps(nan_report, allow_nan=False)


    def test_los_snapshot_passes_web_and_strict_gates(self):
        report = validate_rt_web_snapshot(
            self.tx_settings, self.rt_settings, self.snapshot
        )
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["strict_fd_td_passed"], report)
        self.assertTrue(report["los_oracle"]["passed"], report)
        self.assertGreater(report["reference_channel_power_q_ref"], 0.0)
        json.dumps(report, allow_nan=False)

    def test_snapshot_identity_corruption_fails_closed(self):
        metadata = dict(self.snapshot.metadata)
        metadata["config_sha256"] = "0" * 64
        corrupted = replace(self.snapshot, metadata=metadata)
        report = validate_rt_web_snapshot(
            self.tx_settings, self.rt_settings, corrupted
        )
        self.assertFalse(report["passed"], report)
        identity = next(
            check for check in report["checks"]
            if check["check"] == "snapshot_configuration_software_array_asset_identity"
        )
        self.assertFalse(identity["passed"])
        json.dumps(report, allow_nan=False)

    def test_nonfinite_report_values_become_json_null(self):
        safe = _web_json_safe({"measurement": float("nan"), "array": np.asarray([1.0, float("inf")])})
        self.assertEqual(safe, {"measurement": None, "array": [1.0, None]})
        json.dumps(safe, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
