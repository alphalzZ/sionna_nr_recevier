from dataclasses import replace
from pathlib import Path
import unittest

from nr_pusch.beam_validation import (
    validate_rt_multipath_snapshot,
    validate_rt_path_convergence,
    validate_rt_web_snapshot,
)
from nr_pusch.config import TxSettings
from nr_pusch.rt_channel import prepare_rt_beam_snapshot
from nr_pusch.rt_config import RtBeamSettings


ROOT = Path(__file__).resolve().parents[2]


class RtMultipathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tx_settings = TxSettings.from_toml(ROOT / "configs/pusch_4ue.toml")
        cls.rt_settings = RtBeamSettings.from_toml(ROOT / "configs/rt_beam_ground_wall.toml")
        cls.snapshot = prepare_rt_beam_snapshot(cls.tx_settings, cls.rt_settings)
        cls.report = validate_rt_multipath_snapshot(
            cls.tx_settings,
            cls.rt_settings,
            cls.snapshot,
        )

    def test_scene_materials_are_recorded_at_carrier_frequency(self):
        materials = self.snapshot.metadata["scene_materials"]
        self.assertAlmostEqual(materials["ground-concrete"]["relative_permittivity"], 5.24, places=4)
        self.assertAlmostEqual(materials["ground-concrete"]["conductivity_s_per_m"], 0.123087, places=5)
        self.assertAlmostEqual(materials["wall-brick"]["relative_permittivity"], 3.91, places=4)
        self.assertAlmostEqual(materials["wall-brick"]["conductivity_s_per_m"], 0.0290822, places=6)
        self.assertAlmostEqual(materials["ground-concrete"]["thickness_m"], 0.1, places=6)
        self.assertAlmostEqual(materials["wall-brick"]["thickness_m"], 0.1, places=6)
        self.assertEqual(materials["ground-concrete"]["scattering_coefficient"], 0.0)
        self.assertEqual(materials["wall-brick"]["scattering_coefficient"], 0.0)

    def test_each_ground_and_wall_reflection_matches_independent_oracle(self):
        self.assertTrue(self.report["passed"], self.report)
        self.assertEqual(self.report["path_count_per_user"], [4, 4, 4, 4])
        self.assertEqual(len(self.report["checks"]), 8)
        self.assertEqual(
            {check["material"] for check in self.report["checks"]},
            {"ground-concrete", "wall-brick"},
        )
        for check in self.report["checks"]:
            with self.subTest(user=check["user"], material=check["material"]):
                self.assertLessEqual(check["delay_relative_error"], 1e-5)
                self.assertLessEqual(check["amplitude_relative_error"], 0.02)
                self.assertLessEqual(check["phase_error_rad"], 0.03)


    def test_ground_wall_paths_converge_when_trace_budget_doubles(self):
        higher_settings = replace(
            self.rt_settings,
            rt=replace(self.rt_settings.rt, samples_per_src=200_000),
        )
        higher_snapshot = prepare_rt_beam_snapshot(
            self.tx_settings,
            higher_settings,
        )
        report = validate_rt_path_convergence(self.snapshot, higher_snapshot)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["samples_per_src"], [100_000, 200_000])
        web_report = validate_rt_web_snapshot(
            self.tx_settings,
            self.rt_settings,
            self.snapshot,
            higher_budget_snapshot=higher_snapshot,
        )
        self.assertTrue(web_report["passed"], web_report)
        self.assertFalse(web_report["strict_fd_td_passed"], web_report)
        self.assertFalse(web_report["strict_time_report"]["passed"])
        rms = web_report["strict_time_report"]["fd_td_relative_rms_error"]
        self.assertGreater(rms, 1e-5)
        self.assertLessEqual(rms, 1e-3)
        self.assertTrue(web_report["reflection_oracle"]["passed"])


    def test_convergence_requires_only_a_matching_exactly_doubled_budget(self):
        metadata = dict(self.snapshot.metadata)
        solver = dict(metadata["solver"])
        solver["samples_per_src"] = 150_000
        metadata["solver"] = solver
        wrong_budget = replace(self.snapshot, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "exactly double"):
            validate_rt_path_convergence(self.snapshot, wrong_budget)

        metadata = dict(self.snapshot.metadata)
        metadata["scene_bundle_sha256"] = "different-scene-bundle"
        solver = dict(metadata["solver"])
        solver["samples_per_src"] = 200_000
        metadata["solver"] = solver
        changed_scene = replace(self.snapshot, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "different scene asset bundles"):
            validate_rt_path_convergence(self.snapshot, changed_scene)
if __name__ == "__main__":
    unittest.main()
