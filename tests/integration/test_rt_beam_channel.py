from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import json

import numpy as np
import torch

from nr_pusch.beamforming import user_power_scales
from nr_pusch.config import TxSettings
from nr_pusch.rt_channel import (
    NrPuschRtBeamChannel,
    _cp_energy_diagnostics,
    load_cached_rt_beam_snapshot,
    load_rt_beam_snapshot,
    prepare_rt_beam_snapshot,
    rt_snapshot_cache_key,
    save_rt_beam_snapshot,
    store_rt_beam_snapshot_cache,
)
from nr_pusch.rt_config import RtBeamSettings
from nr_pusch.rt_scene_assets import (
    copy_scene_assets,
    make_scene_template_zip,
    parse_scene_bundle_zip,
    persist_scene_bundle,
    resolve_builtin_scene_assets,
)
from nr_pusch.transmitter import NrPuschTx
from nr_pusch.beam_validation import (
    rt_tap_frequency_response,
    validate_rt_beam_structure,
    validate_rt_los_snapshot,
    validate_rt_time_domain,
    validate_rt_uncoded,
    validate_rt_web_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]


class RtBeamChannelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tx_settings = TxSettings.from_toml(ROOT / "configs/pusch_4ue.toml")
        cls.rt_settings = RtBeamSettings.from_toml(ROOT / "configs/rt_beam_los.toml")
        cls.snapshot = prepare_rt_beam_snapshot(cls.tx_settings, cls.rt_settings)

    def test_los_delay_friis_phase_and_synthetic_array_response(self):
        bs = np.asarray(self.rt_settings.receiver.position_m, dtype=np.float64)
        fc = self.rt_settings.rt.carrier_frequency_hz
        wavelength = 299_792_458.0 / fc
        for user_index, user in enumerate(self.rt_settings.users):
            displacement = np.asarray(user.position_m, dtype=np.float64) - bs
            distance = float(np.linalg.norm(displacement))
            direction = displacement / distance
            delays = self.snapshot.path_tau_s[:, user_index, 0]
            np.testing.assert_allclose(delays * 299_792_458.0, distance, rtol=1e-5, atol=1e-6)

            array_phase = np.exp(
                2j * np.pi * (self.snapshot.element_positions_m @ direction) / wavelength
            )
            coefficients = self.snapshot.path_a[:, user_index, 0]
            center_coefficient = np.mean(coefficients * array_phase.conj())
            reconstructed = center_coefficient * array_phase
            relative_spatial_error = np.linalg.norm(coefficients - reconstructed) / np.linalg.norm(coefficients)
            self.assertLessEqual(float(relative_spatial_error), 1e-3)

            friis_amplitude = wavelength / (4.0 * np.pi * distance)
            self.assertLessEqual(
                abs(abs(center_coefficient) ** 2 / friis_amplitude**2 - 1.0),
                0.01,
            )
            analytic = friis_amplitude * np.exp(-2j * np.pi * fc * distance / 299_792_458.0)
            self.assertLessEqual(abs(center_coefficient / analytic - 1.0), 1e-3)
        self.assertEqual(self.snapshot.metadata["valid_path_count_per_user"], [1, 1, 1, 1])

        self.assertLessEqual(self.snapshot.metadata["path_cfr_relative_error_after_common_ta"], 1e-5)
        self.assertTrue(self.snapshot.metadata["cp_sufficient"])

    def test_frequency_and_time_adapters_preserve_user_contributions(self):
        tx = NrPuschTx(self.tx_settings, device="cpu")
        generated = tx.generate(batch_size=1, seed=13)
        channel = NrPuschRtBeamChannel(self.snapshot, device="cpu")
        frequency = channel.apply_frequency(generated.frequency_grid, tx._tx_freq.resource_grid)
        power = torch.as_tensor(
            np.sqrt(user_power_scales(self.snapshot.power_scale_db)),
            dtype=torch.float32,
        )
        h_beam = torch.as_tensor(self.snapshot.h_beam, dtype=torch.complex64)
        expected_per_user = (
            generated.frequency_grid[:, :, 0, :, :][:, :, None, :, :]
            * (h_beam * power[None, :, None]).permute(1, 0, 2)[None, :, :, None, :]
        )
        torch.testing.assert_close(frequency.per_user_grid, expected_per_user)
        torch.testing.assert_close(
            frequency.grid,
            expected_per_user.sum(dim=1).unsqueeze(1),
        )
        self.assertEqual(tuple(frequency.grid.shape), (1, 1, 4, 14, 600))
        self.assertEqual(tuple(frequency.channel_frequency_response.shape), (1, 1, 4, 4, 1, 14, 600))

        time = channel.apply(generated.iq, generated.sample_rate_hz)
        self.assertEqual(time.iq.shape[:2], (1, 4))
        self.assertEqual(time.per_user_iq.shape[:3], (1, 4, 4))
        self.assertEqual(time.iq.shape[-1], generated.iq.shape[-1] + time.metadata["channel_tail_samples"])
        torch.testing.assert_close(time.iq, time.per_user_iq.sum(dim=1))
        self.assertEqual(time.channel_taps.shape[:4], (1, 4, 4, 1))

    def test_los_and_shared_beam_physics_validators(self):
        los_report = validate_rt_los_snapshot(
            self.tx_settings,
            self.rt_settings,
            self.snapshot,
        )
        beam_report = validate_rt_beam_structure(self.snapshot)

        self.assertTrue(los_report["passed"], los_report)
        self.assertTrue(beam_report["passed"], beam_report)

    def test_uncoded_baselines_match_analytical_results(self):
        report = validate_rt_uncoded(self.snapshot, seed=13)
        self.assertTrue(report["passed"], report)

    def test_time_domain_validation_matches_native_frequency_grid(self):
        report = validate_rt_time_domain(
            self.tx_settings,
            self.snapshot,
            device="cpu",
            seed=13,
        )

        self.assertTrue(report["passed"], report)
        self.assertTrue(report["cp_sufficient"])
        self.assertEqual(report["fd_td_status"], "passed")
        self.assertLessEqual(report["fd_td_relative_rms_error"], 1e-5)
        self.assertLessEqual(report["direct_rt_cfr_truncation_relative_error"], 0.01)
        self.assertEqual(
            len(report["cp_energy_interval_samples_by_beam_user"]),
            4,
        )
        self.assertEqual(
            len(report["cp_energy_interval_bounds_samples_by_beam_user"]),
            4,
        )

    def test_web_gate_reports_conditional_los_acceptance(self):
        report = validate_rt_web_snapshot(
            self.tx_settings, self.rt_settings, self.snapshot
        )
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["strict_fd_td_passed"], report)
        self.assertTrue(report["los_oracle"]["passed"], report)

    def test_time_gate_rejects_low_energy_echo_beyond_cp(self):
        taps = self.snapshot.taps_beam.copy()
        taps[0, 0, -1] += 0.002 * float(np.max(np.abs(taps)))
        metadata = dict(self.snapshot.metadata)
        spans, bounds, outside, sufficient = _cp_energy_diagnostics(
            taps,
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
        echo_snapshot = replace(self.snapshot, taps_beam=taps, metadata=metadata)
        echo_snapshot = replace(
            echo_snapshot,
            h_beam=rt_tap_frequency_response(echo_snapshot),
        )

        report = validate_rt_time_domain(
            self.tx_settings,
            echo_snapshot,
            device="cpu",
            seed=13,
        )

        self.assertTrue(report["cp_sufficient"])
        self.assertLess(metadata["cp_outside_energy_ratio_max"], 0.01)
        self.assertEqual(report["fd_td_status"], "failed")
        self.assertGreater(report["fd_td_relative_rms_error"], 1e-5)
        self.assertFalse(report["passed"])

    def test_snapshot_roundtrip_allows_mcs_but_rejects_dmrs_geometry_change(self):
        self.assertEqual(self.snapshot.metadata["format_version"], 2)
        self.assertEqual(
            set(self.snapshot.metadata["scene_asset_sha256"]),
            {"empty.xml"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "snapshot.npz"
            save_rt_beam_snapshot(self.snapshot, path)
            loaded = load_rt_beam_snapshot(path, self.tx_settings)
            np.testing.assert_array_equal(loaded.h_beam, self.snapshot.h_beam)

            job_assets = copy_scene_assets(
                resolve_builtin_scene_assets("empty"), root / "scene"
            )
            replay = load_rt_beam_snapshot(
                path, self.tx_settings, scene_root=job_assets.root
            )
            np.testing.assert_array_equal(replay.h_beam, self.snapshot.h_beam)

            changed_mcs = replace(
                self.tx_settings,
                pusch=replace(self.tx_settings.pusch, mcs_index=self.tx_settings.pusch.mcs_index - 1),
            )
            mcs_replay = load_rt_beam_snapshot(path, changed_mcs)
            np.testing.assert_array_equal(mcs_replay.h_beam, self.snapshot.h_beam)

            changed_dmrs = replace(
                self.tx_settings,
                pusch=replace(
                    self.tx_settings.pusch,
                    dmrs_additional_position=(
                        0 if self.tx_settings.pusch.dmrs_additional_position != 0 else 1
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "resource-grid/DMRS geometry"):
                load_rt_beam_snapshot(path, changed_dmrs)

            manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            manifest["format_version"] = 1
            path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重新 prepare"):
                load_rt_beam_snapshot(path, self.tx_settings)

    def test_snapshot_cache_reuses_verified_arrays_and_invalidates_rt_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "rt_snapshots"
            scene_assets = resolve_builtin_scene_assets("empty")
            key = store_rt_beam_snapshot_cache(
                self.snapshot,
                self.tx_settings,
                self.rt_settings,
                scene_assets=scene_assets,
                cache_dir=cache_dir,
            )
            cached, cached_key = load_cached_rt_beam_snapshot(
                self.tx_settings,
                self.rt_settings,
                scene_assets=scene_assets,
                cache_dir=cache_dir,
            )
            self.assertIsNotNone(cached)
            self.assertEqual(key, cached_key)
            np.testing.assert_array_equal(cached.h_ant, self.snapshot.h_ant)
            self.assertTrue((cache_dir / key / "cache.json").is_file())
            cached_npz = cache_dir / key / "channel_snapshot.npz"
            cached_npz.write_bytes(b"not-a-zip")
            corrupted, _ = load_cached_rt_beam_snapshot(
                self.tx_settings,
                self.rt_settings,
                scene_assets=scene_assets,
                cache_dir=cache_dir,
            )
            self.assertIsNone(corrupted)

            changed = replace(
                self.rt_settings,
                rt=replace(self.rt_settings.rt, seed=self.rt_settings.rt.seed + 1),
            )
            changed_key, _ = rt_snapshot_cache_key(
                self.tx_settings,
                changed,
                scene_assets=scene_assets,
            )
            stale, _ = load_cached_rt_beam_snapshot(
                self.tx_settings,
                changed,
                scene_assets=scene_assets,
                cache_dir=cache_dir,
            )
            self.assertNotEqual(key, changed_key)
            self.assertIsNone(stale)

    def test_custom_scene_assets_prepare_save_and_replay_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = persist_scene_bundle(
                parse_scene_bundle_zip(make_scene_template_zip()),
                root / "uploaded",
            )
            custom_settings = replace(
                self.rt_settings,
                rt=replace(
                    self.rt_settings.rt,
                    scene="custom",
                    scene_file="scene.xml",
                    max_depth=2,
                    samples_per_src=10_000,
                ),
            )
            snapshot = prepare_rt_beam_snapshot(
                self.tx_settings, custom_settings, scene_assets=assets
            )
            path = root / "custom.npz"
            save_rt_beam_snapshot(snapshot, path)
            self.assertEqual(snapshot.metadata["scene_source"], "imported")
            self.assertEqual(snapshot.metadata["scene_file"], "scene.xml")
            with self.assertRaisesRegex(ValueError, "scene_root"):
                load_rt_beam_snapshot(path, self.tx_settings)
            loaded = load_rt_beam_snapshot(
                path, self.tx_settings, scene_root=assets.root
            )
            np.testing.assert_array_equal(loaded.h_beam, snapshot.h_beam)

            ground = assets.root / "meshes" / "ground.ply"
            ground.write_bytes(ground.read_bytes().replace(b"-500.0 -500.0 0.0", b"-500.00 -500.0 0.0", 1))
            with self.assertRaisesRegex(ValueError, "场景资源 hash"):
                load_rt_beam_snapshot(
                    path, self.tx_settings, scene_root=assets.root
                )



if __name__ == "__main__":
    unittest.main()
