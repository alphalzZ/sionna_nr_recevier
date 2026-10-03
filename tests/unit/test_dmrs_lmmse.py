from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import shutil
import unittest

from nr_pusch.channel_config import ChannelSettings
from nr_pusch.dmrs_prior import (
    default_prior_dir,
    dmrs_prior_compatibility,
    load_dmrs_tap_power_prior,
    prior_registry_path,
    publish_accepted_dmrs_prior,
    resolve_dmrs_tap_power_prior,
    save_dmrs_tap_power_prior,
)

import torch
import sionna.phy

from nr_pusch.config import TxSettings
from nr_pusch.receiver import DftSOfdmDmrsEstimator, NrPuschRx
from nr_pusch.transmitter import NrPuschTx


ROOT = Path(__file__).parents[2]


class DmrsTapLmmseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sionna.phy.config.seed = 17
        cls.settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        cls.transmitter = NrPuschTx(cls.settings, device="cpu")
        cls.tx_result = cls.transmitter.generate(batch_size=1, seed=17)
        cls.resource_grid = cls.transmitter._tx_freq.resource_grid
        mask = cls.transmitter._tx_freq.pilot_pattern.mask[0, 0]
        cls.dmrs_symbol = int(torch.where(mask.sum(dim=-1) > 0)[0].item())
        cls.pilot_grid = cls.tx_result.frequency_grid[0, :, 0, cls.dmrs_symbol, :]
        cls.l_min = -1
        cls.l_max = 2
        cls.prior = torch.tensor([0.8, 0.5, 0.2, 0.1], dtype=torch.float32)

    def _estimator(self, prior=None):
        return DftSOfdmDmrsEstimator(
            self.pilot_grid,
            self.resource_grid,
            self.l_min,
            self.l_max,
            tap_power_prior=prior,
        )

    def _observations(self, estimator):
        y = torch.zeros(
            (1, 1, 1, self.resource_grid.num_ofdm_symbols, self.resource_grid.fft_size),
            dtype=self.pilot_grid.dtype,
        )
        first_rhs = None
        first_design = None
        for pair_index, (_, _, design_name, support_name, _) in enumerate(estimator._pairs):
            design = getattr(estimator, design_name)
            support = getattr(estimator, support_name)
            coefficients = torch.arange(1, design.shape[1] + 1, dtype=torch.float32)
            coefficients = coefficients.to(dtype=design.dtype) * (0.1 + 0.03j * (pair_index + 1))
            noise = torch.arange(design.shape[0], dtype=torch.float32)
            noise = (0.02 * torch.cos(noise) + 0.015j * torch.sin(noise)).to(design.dtype)
            observations = design @ coefficients + noise
            y[0, 0, 0, self.dmrs_symbol, support] = observations
            if pair_index == 0:
                first_rhs = observations
                first_design = design
        return y, first_design, first_rhs

    def test_lmmse_mean_posterior_projection_and_zero_noise_ls(self):
        estimator = self._estimator(self.prior)
        y, design, rhs = self._observations(estimator)
        noise_variance = 0.2
        h_hat, err_var = estimator(y, noise_variance)

        tap_power = torch.cat((self.prior, self.prior)).to(dtype=design.real.dtype)
        prior_covariance = torch.diag(tap_power.to(dtype=design.dtype))
        system = design @ prior_covariance @ design.mH
        system += noise_variance * torch.eye(design.shape[0], dtype=design.dtype)
        expected_taps = prior_covariance @ design.mH @ torch.linalg.solve(system, rhs)
        posterior = prior_covariance - prior_covariance @ design.mH @ torch.linalg.solve(
            system, design @ prior_covariance
        )
        frequency_basis = estimator._frequency_basis
        expected_frequency = expected_taps[: self.prior.numel()] @ frequency_basis.T
        expected_error = torch.einsum(
            "nl,lm,nm->n",
            frequency_basis,
            posterior[: self.prior.numel(), : self.prior.numel()],
            frequency_basis.conj(),
        ).real.clamp_min(0.0)

        self.assertEqual(tuple(h_hat.shape), (1, 1, 1, 4, 1, 14, self.resource_grid.fft_size))
        self.assertEqual(tuple(err_var.shape), tuple(h_hat.shape[:-1]) + (self.resource_grid.fft_size,))
        torch.testing.assert_close(
            h_hat[0, 0, 0, estimator._pairs[0][0], 0, 0], expected_frequency
        )
        torch.testing.assert_close(
            err_var[0, 0, 0, estimator._pairs[0][0], 0, 0], expected_error
        )
        torch.testing.assert_close(
            err_var[0, 0, 0, estimator._pairs[0][0], 0],
            expected_error.expand(self.resource_grid.num_ofdm_symbols, -1),
        )

        h_zero, err_zero = estimator(y, 0.0)
        expected_ls = torch.linalg.lstsq(design, rhs).solution[: self.prior.numel()]
        expected_ls_frequency = expected_ls @ frequency_basis.T
        torch.testing.assert_close(
            h_zero[0, 0, 0, estimator._pairs[0][0], 0, 0], expected_ls_frequency
        )
        self.assertTrue(torch.equal(err_zero, torch.zeros_like(err_zero)))
        with self.assertRaisesRegex(ValueError, "noise variance"):
            estimator(y, -0.1)

    def test_prior_artifact_round_trips_and_rejects_incompatible_geometry(self):
        channel_settings = ChannelSettings.from_toml(
            ROOT / "configs" / "cdl_38_901_4x4.toml"
        )
        compatibility = dmrs_prior_compatibility(
            self.settings,
            channel_settings,
            l_min=self.l_min,
            max_delay_spread_s=3e-6,
            fft_size=self.resource_grid.fft_size,
            sample_rate_hz=self.transmitter.sample_rate_hz,
        )
        taps = compatibility["l_max"] - compatibility["l_min"] + 1
        with tempfile.TemporaryDirectory() as directory:
            prior_path = Path(directory) / "prior.npz"
            prior = torch.full((taps,), 0.25)
            save_dmrs_tap_power_prior(
                prior_path,
                prior,
                compatibility=compatibility,
                training_seed=42,
                training_realizations=8,
            )
            loaded, metadata = load_dmrs_tap_power_prior(
                prior_path, expected_compatibility=compatibility
            )
            torch.testing.assert_close(loaded, prior)
            self.assertEqual(metadata["training_seed"], 42)
            legacy_compatibility = {
                **compatibility,
                "tx_pusch": {
                    **compatibility["tx_pusch"],
                    "mcs_table": self.settings.pusch.mcs_table,
                    "mcs_index": self.settings.pusch.mcs_index,
                },
            }
            legacy_prior_path = Path(directory) / "legacy_prior.npz"
            save_dmrs_tap_power_prior(
                legacy_prior_path,
                prior,
                compatibility=legacy_compatibility,
                training_seed=42,
                training_realizations=8,
            )
            legacy_prior, _ = load_dmrs_tap_power_prior(
                legacy_prior_path, expected_compatibility=compatibility
            )
            torch.testing.assert_close(legacy_prior, prior)
            mismatch = {**compatibility, "l_min": compatibility["l_min"] - 1}
            with self.assertRaisesRegex(ValueError, "不兼容"):
                load_dmrs_tap_power_prior(
                    prior_path, expected_compatibility=mismatch
                )
            cp_settings = replace(
                self.settings,
                pusch=replace(
                    self.settings.pusch,
                    waveform="cp_ofdm",
                    dmrs_beta=2**0.5,
                ),
            )
            cp_settings.validate()
            cp_compatibility = dmrs_prior_compatibility(
                cp_settings,
                channel_settings,
                l_min=self.l_min,
                max_delay_spread_s=3e-6,
                fft_size=self.resource_grid.fft_size,
                sample_rate_hz=self.transmitter.sample_rate_hz,
            )
            self.assertEqual(cp_compatibility["tx_pusch"]["waveform"], "cp_ofdm")
            cp_prior_path = Path(directory) / "cp_prior.npz"
            save_dmrs_tap_power_prior(
                cp_prior_path,
                prior,
                compatibility=cp_compatibility,
                training_seed=42,
                training_realizations=8,
            )
            with self.assertRaisesRegex(ValueError, "不兼容"):
                load_dmrs_tap_power_prior(
                    prior_path, expected_compatibility=cp_compatibility
                )
            with self.assertRaisesRegex(ValueError, "不兼容"):
                load_dmrs_tap_power_prior(
                    cp_prior_path, expected_compatibility=compatibility
                )

    def test_lmmse_requires_valid_prior(self):
        with self.assertRaisesRegex(ValueError, "必须提供 dmrs_tap_power_prior"):
            NrPuschRx(self.settings, channel_estimator="dmrs-lmmse", device="cpu")
        for prior in (
            torch.ones(3),
            torch.tensor([1.0, 1.0, 0.0, 1.0]),
            torch.tensor([1.0, float("nan"), 1.0, 1.0]),
        ):
            with self.subTest(prior=prior):
                with self.assertRaises(ValueError):
                    self._estimator(prior)
        with self.assertRaisesRegex(ValueError, "仅能用于 dmrs-lmmse"):
            NrPuschRx(
                self.settings,
                channel_estimator="perfect",
                dmrs_tap_power_prior=self.prior,
                device="cpu",
            )



class DmrsPriorRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sionna.phy.config.seed = 17
        cls.settings = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")
        cls.transmitter = NrPuschTx(cls.settings, device="cpu")
        cls.channel_settings = ChannelSettings.from_toml(
            ROOT / "configs" / "cdl_38_901_4x4.toml"
        )

    def _compatibility(self, **overrides):
        values = dict(
            l_min=-6,
            max_delay_spread_s=3e-6,
            fft_size=self.transmitter._tx_freq.resource_grid.fft_size,
            sample_rate_hz=self.transmitter.sample_rate_hz,
        )
        values.update(overrides)
        return dmrs_prior_compatibility(
            self.settings, self.channel_settings, **values
        )

    def _candidate(self, directory: Path, compatibility) -> Path:
        taps = compatibility["l_max"] - compatibility["l_min"] + 1
        path = Path(directory) / "candidate.prior.npz"
        save_dmrs_tap_power_prior(
            path,
            torch.full((taps,), 0.25),
            compatibility=compatibility,
            training_seed=42,
            training_realizations=8,
        )
        return path

    def test_registry_key_separates_waveforms_and_ignores_mcs(self):
        dft = self._compatibility()
        cp_settings = replace(
            self.settings,
            pusch=replace(self.settings.pusch, waveform="cp_ofdm", dmrs_beta=2**0.5),
        )
        cp_settings.validate()
        cp = dmrs_prior_compatibility(
            cp_settings,
            self.channel_settings,
            l_min=-6,
            max_delay_spread_s=3e-6,
            fft_size=self.transmitter._tx_freq.resource_grid.fft_size,
            sample_rate_hz=self.transmitter.sample_rate_hz,
        )
        store = Path("configs") / "tap_power_prior"
        self.assertNotEqual(prior_registry_path(store, dft), prior_registry_path(store, cp))
        self.assertEqual(
            prior_registry_path(store, dft).parent.name,
            "dft_s_ofdm",
        )
        self.assertEqual(
            prior_registry_path(store, cp).parent.name,
            "cp_ofdm",
        )
        mcs_changed = replace(
            self.settings,
            pusch=replace(self.settings.pusch, mcs_index=1),
        )
        mcs_changed.validate()
        self.assertEqual(
            prior_registry_path(
                store,
                dmrs_prior_compatibility(
                    mcs_changed,
                    self.channel_settings,
                    l_min=-6,
                    max_delay_spread_s=3e-6,
                    fft_size=self.transmitter._tx_freq.resource_grid.fft_size,
                    sample_rate_hz=self.transmitter.sample_rate_hz,
                ),
            ),
            prior_registry_path(store, dft),
        )
        self.assertNotEqual(
            prior_registry_path(store, self._compatibility(max_delay_spread_s=2e-6)),
            prior_registry_path(store, dft),
        )

    def test_publish_requires_an_accepted_gate_and_resolves_by_key(self):
        compatibility = self._compatibility()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "tap_power_prior"
            candidate = self._candidate(root, compatibility)
            with self.assertRaisesRegex(ValueError, "acceptance_gate"):
                publish_accepted_dmrs_prior(
                    store,
                    candidate,
                    compatibility=compatibility,
                    acceptance_gate={"passed": False},
                )
            self.assertFalse(store.exists())
            published = publish_accepted_dmrs_prior(
                store,
                candidate,
                compatibility=compatibility,
                acceptance_gate={"passed": True, "margin_db": 1.25},
            )
            self.assertEqual(published, prior_registry_path(store, compatibility))
            prior, path, metadata = resolve_dmrs_tap_power_prior(
                store, expected_compatibility=compatibility
            )
            self.assertEqual(path, published)
            self.assertEqual(prior.shape, (compatibility["l_max"] + 7,))
            self.assertRegex(metadata["accepted_gate_sha256"], r"\A[0-9a-f]{64}\Z")
            self.assertEqual(metadata["training_realizations"], 8)
            with self.assertRaisesRegex(ValueError, "未找到"):
                resolve_dmrs_tap_power_prior(
                    store, expected_compatibility=self._compatibility(l_min=-5)
                )
            self.assertEqual(list(store.rglob("*.tmp")), [])

    def test_unaccepted_candidate_is_not_selectable_and_republish_is_atomic(self):
        compatibility = self._compatibility()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "tap_power_prior"
            candidate = self._candidate(root, compatibility)
            unmarked = prior_registry_path(store, compatibility)
            unmarked.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, unmarked)
            with self.assertRaisesRegex(ValueError, "缺少通过验证的 acceptance 标记"):
                resolve_dmrs_tap_power_prior(
                    store, expected_compatibility=compatibility
                )
            published = publish_accepted_dmrs_prior(
                store,
                candidate,
                compatibility=compatibility,
                acceptance_gate={"passed": True},
            )
            replacement = self._candidate(root, compatibility)
            with self.assertRaises(ValueError):
                publish_accepted_dmrs_prior(
                    store,
                    replacement,
                    compatibility=self._compatibility(max_delay_spread_s=2e-6),
                    acceptance_gate={"passed": True},
                )
            prior, _, _ = resolve_dmrs_tap_power_prior(
                store, expected_compatibility=compatibility
            )
            torch.testing.assert_close(prior, torch.full((compatibility["l_max"] + 7,), 0.25))
            self.assertTrue(published.is_file())
            self.assertEqual(list(store.rglob("*.tmp")), [])

    def test_default_prior_dir_sits_beside_the_channel_profile(self):
        self.assertEqual(
            default_prior_dir(ROOT / "configs" / "cdl_38_901_4x4.toml"),
            ROOT / "configs" / "tap_power_prior",
        )

if __name__ == "__main__":
    unittest.main()
