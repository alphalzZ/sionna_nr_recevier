from pathlib import Path
import tomllib
import unittest

from nr_pusch.beam_validation import (
    _bler_settings_toml,
    single_layer_slab_reflection_coefficients,
)
from nr_pusch.rt_config import RtBeamSettings
from nr_pusch.simulation_config import BlerSettings

class SlabReflectionTest(unittest.TestCase):
    def test_zero_thickness_has_no_reflection(self):
        te, tm = single_layer_slab_reflection_coefficients(
            frequency_hz=3.5e9,
            thickness_m=0.0,
            relative_permittivity=5.24,
            conductivity_s_m=0.123087,
            cos_incidence=0.7,
        )
        self.assertAlmostEqual(abs(te), 0.0, places=14)
        self.assertAlmostEqual(abs(tm), 0.0, places=14)

    def test_normal_incidence_has_opposite_te_tm_signs_for_finite_slab(self):
        te, tm = single_layer_slab_reflection_coefficients(
            frequency_hz=3.5e9,
            thickness_m=0.1,
            relative_permittivity=5.24,
            conductivity_s_m=0.123087,
            cos_incidence=1.0,
        )
        self.assertGreater(abs(te), 0.0)
        self.assertLess(abs(te + tm), 1e-12)

    def test_rejects_non_finite_or_out_of_domain_inputs(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            single_layer_slab_reflection_coefficients(
                frequency_hz=float("nan"),
                thickness_m=0.1,
                relative_permittivity=5.24,
                conductivity_s_m=0.1,
                cos_incidence=0.7,
            )
        with self.assertRaisesRegex(ValueError, "incidence cosine"):
            single_layer_slab_reflection_coefficients(
                frequency_hz=3.5e9,
                thickness_m=0.1,
                relative_permittivity=5.24,
                conductivity_s_m=0.1,
                cos_incidence=1.1,
            )

class ResolvedValidationTomlTest(unittest.TestCase):
    def test_resolved_rt_and_bler_settings_remain_parseable_toml(self):
        root = Path(__file__).resolve().parents[2]
        rt_settings = RtBeamSettings.from_toml(root / "configs/rt_beam_los.toml")
        simulation_settings = BlerSettings.from_toml(
            root / "configs/bler_rt_beam_smoke.toml"
        )

        rt = tomllib.loads(rt_settings.to_toml())
        bler = tomllib.loads(_bler_settings_toml(simulation_settings))["bler"]

        self.assertEqual(rt["rt"]["samples_per_src"], rt_settings.rt.samples_per_src)
        self.assertEqual(
            [user["name"] for user in rt["users"]],
            [user.name for user in rt_settings.users],
        )
        self.assertEqual(bler["max_frames_per_snr"], simulation_settings.max_frames_per_snr)
        self.assertEqual(bler["detectors"], list(simulation_settings.detectors))


if __name__ == "__main__":
    unittest.main()
