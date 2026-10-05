from dataclasses import replace
from pathlib import Path
import tempfile
import tomllib
import unittest

from nr_pusch.config import TxSettings
from nr_pusch.rt_config import RtBeamSettings
from nr_pusch.rt_scene_assets import SIONNA_RT_SCENE_IDS


ROOT = Path(__file__).parents[2]


class RtBeamSettingsTest(unittest.TestCase):
    def test_loads_all_scene_profiles_and_matches_four_user_tx_order(self):
        tx = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        expected = {"rt_beam_los.toml": ("empty", 0), "rt_beam_ground.toml": ("ground", 1), "rt_beam_ground_wall.toml": ("ground_wall", 2)}
        for filename, (scene_name, depth) in expected.items():
            with self.subTest(profile=filename):
                settings = RtBeamSettings.from_toml(ROOT / "configs" / filename)
                settings.validate_transmitter(tx)
                self.assertEqual((settings.rt.scene, settings.rt.max_depth), (scene_name, depth))
                self.assertEqual(tuple(user.name for user in settings.users), tuple(user.name for user in tx.users))
                self.assertEqual(settings.receiver.num_rows * settings.receiver.num_cols, 64)
                self.assertEqual(settings.noise.post_combiner_ratio, 0.001)

    def test_rejects_unknown_root_and_nested_fields(self):
        source = (ROOT / "configs" / "rt_beam_los.toml").read_text(encoding="utf-8")
        for modified, expected in (
            ("unsupported = true\n" + source, "root 包含未知字段"),
            (source.replace('scene = "empty"', 'scene = "empty"\nunknown_solver_flag = true'), "rt 包含未知字段"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.toml"
                path.write_text(modified, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    RtBeamSettings.from_toml(path)

    def test_rejects_nonfinite_values_wrong_vectors_and_noise_ratio(self):
        source = (ROOT / "configs" / "rt_beam_los.toml").read_text(encoding="utf-8")
        invalid_profiles = (
            (source.replace("carrier_frequency_hz = 3500000000.0", "carrier_frequency_hz = nan"), "有限值"),
            (source.replace("position_m = [153.2088886, -128.5575220, 1.5]", "position_m = [1.0, 2.0]", 1), "3 个"),
            (source.replace("post_combiner_ratio = 0.001", "post_combiner_ratio = 0.02"), "\\[0,0.01\\]"),
        )
        for profile, expected in invalid_profiles:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.toml"
                path.write_text(profile, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    RtBeamSettings.from_toml(path)

    def test_rejects_multilayer_and_mismatched_user_order(self):
        settings = RtBeamSettings.from_toml(ROOT / "configs" / "rt_beam_los.toml")
        multilayer = TxSettings.from_toml(ROOT / "configs" / "pusch_cp_2ue_2layer.toml")
        with self.assertRaisesRegex(ValueError, "恰好四个"):
            settings.validate_transmitter(multilayer)

        tx = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        reordered = replace(tx, users=(tx.users[1], tx.users[0], tx.users[2], tx.users[3]))
        with self.assertRaisesRegex(ValueError, "原顺序完全匹配"):
            settings.validate_transmitter(reordered)

    def test_dict_and_toml_round_trip_parameterized_geometry(self):
        source = RtBeamSettings.from_toml(ROOT / "configs" / "rt_beam_ground_wall.toml")
        raw = source.to_dict()
        raw["geometry"] = {
            "ground_bounds_m": [-20.0, 40.0, -30.0, 50.0],
            "ground_height_m": 1.25,
            "ground_material": "brick",
            "ground_thickness_m": 0.2,
            "wall_start_xy_m": [-5.0, 12.0],
            "wall_end_xy_m": [8.0, 16.0],
            "wall_base_height_m": 1.25,
            "wall_height_m": 8.0,
            "wall_material": "concrete",
            "wall_thickness_m": 0.3,
        }
        settings = RtBeamSettings.from_dict(raw)
        reparsed = RtBeamSettings.from_dict(tomllib.loads(settings.to_toml()))
        self.assertEqual(reparsed, settings)
        self.assertEqual(settings.geometry.ground_bounds_m, (-20.0, 40.0, -30.0, 50.0))
        self.assertEqual(settings.geometry.wall_material, "concrete")

    def test_accepts_every_sionna_scene_and_rejects_parameterized_geometry(self):
        source = RtBeamSettings.from_toml(ROOT / "configs" / "rt_beam_los.toml")
        for scene_id in sorted(SIONNA_RT_SCENE_IDS):
            with self.subTest(scene=scene_id):
                raw = source.to_dict()
                raw["rt"]["scene"] = scene_id
                settings = RtBeamSettings.from_dict(raw)
                self.assertEqual(settings.rt.scene, scene_id)
                with self.assertRaisesRegex(ValueError, "仅适用于 ground 或 ground_wall"):
                    RtBeamSettings.from_dict(raw | {"geometry": {}})

    def test_scene_file_and_geometry_scope_are_enforced(self):
        source = RtBeamSettings.from_toml(ROOT / "configs" / "rt_beam_los.toml")
        raw = source.to_dict()
        raw["rt"]["scene"] = "custom"
        with self.assertRaisesRegex(ValueError, "scene_file"):
            RtBeamSettings.from_dict(raw)
        raw["rt"]["scene_file"] = "scene.xml"
        with self.assertRaisesRegex(ValueError, "仅适用于 ground"):
            RtBeamSettings.from_dict(raw | {"geometry": {}})
        raw["rt"]["scene"] = "ground"
        raw["rt"]["scene_file"] = None
        with self.assertRaisesRegex(ValueError, "不接受 wall geometry"):
            RtBeamSettings.from_dict(raw | {"geometry": {"wall_height_m": 2.0}})

    def test_rejects_geometry_out_of_range_and_nonfinite(self):
        source = RtBeamSettings.from_toml(ROOT / "configs" / "rt_beam_ground_wall.toml")
        raw = source.to_dict()
        raw["geometry"] = {"ground_bounds_m": [2.0, 1.0, -1.0, 1.0]}
        with self.assertRaisesRegex(ValueError, "严格递增"):
            RtBeamSettings.from_dict(raw)
        raw["geometry"] = {"wall_height_m": float("inf")}
        with self.assertRaisesRegex(ValueError, "有限值"):
            RtBeamSettings.from_dict(raw)
        raw["geometry"] = {"wall_base_height_m": 9_000.0, "wall_height_m": 2_000.0}
        with self.assertRaisesRegex(ValueError, "坐标绝对值"):
            RtBeamSettings.from_dict(raw)

    def test_web_limits_bound_resources_and_report_all_exceeded_fields(self):
        from nr_pusch.rt_config import validate_rt_web_limits
        from nr_pusch.simulation_config import BlerSettings

        settings = RtBeamSettings.from_toml(ROOT / "configs" / "rt_beam_los.toml")
        quick = BlerSettings.from_toml(ROOT / "configs" / "bler_rt_beam_web_quick.toml")
        validate_rt_web_limits(settings, quick)
        long = BlerSettings.from_toml(ROOT / "configs" / "bler_rt_beam.toml")
        validate_rt_web_limits(settings, long)
        self.assertEqual(long.batch_size, 20)
        self.assertEqual(long.max_frames_per_snr, 2_000)
        self.assertTrue(long.stop_at_zero_bler)

        invalid = replace(long, batch_size=21, max_frames_per_snr=2_001)
        with self.assertRaisesRegex(
            ValueError,
            r"RT Web limits exceeded:.*bler\.batch_size=21.*bler\.max_frames_per_snr=2001",
        ):
            validate_rt_web_limits(settings, invalid)

        oversized = replace(settings, receiver=replace(settings.receiver, num_rows=16))
        with self.assertRaisesRegex(ValueError, r"receiver array elements=128"):
            validate_rt_web_limits(oversized, quick)

    def test_cp_rt_profile_preserves_four_user_single_layer_grid(self):
        tx = TxSettings.from_toml(ROOT / "configs" / "pusch_rt_4ue_cp.toml")
        settings = RtBeamSettings.from_toml(ROOT / "configs" / "rt_beam_los.toml")
        settings.validate_transmitter(tx)
        self.assertEqual(tx.pusch.waveform, "cp_ofdm")
        self.assertEqual(tx.pusch.mcs_index, 20)
        self.assertEqual(tuple(user.dmrs_ports for user in tx.users), ((0,), (1,), (2,), (3,)))


if __name__ == "__main__":
    unittest.main()
