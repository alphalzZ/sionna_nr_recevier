"""Web API coverage for estimator-detector matrix runs."""

from http.server import ThreadingHTTPServer
import json
import numpy as np
from pathlib import Path
import shutil
import tempfile
from threading import Event, Thread
from urllib.request import Request, urlopen
import unittest

from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.dmrs_prior import (
    dmrs_prior_compatibility,
    prior_registry_path,
    publish_accepted_dmrs_prior,
    save_dmrs_tap_power_prior,
)
from nr_pusch.simulation_config import BlerSettings
from nr_pusch.transmitter import NrPuschTx
from nr_pusch.web import ApiError, SimulationWebApp, make_handler

ROOT = Path(__file__).parents[2]
TX_CONFIG = ROOT / "configs" / "pusch_4ue.toml"
CHANNEL_CONFIG = ROOT / "configs" / "cdl_38_901_4x4.toml"
SIMULATION_CONFIG = ROOT / "configs" / "bler_estimator_matrix.toml"


SCENARIO_IDS = (
    "dft-4ue-cdl-a",
    "dft-4ue-cdl-b",
    "dft-4ue-cdl-c",
    "dft-4ue-cdl-d",
    "dft-4ue-cdl-e",
    "dft-4ue-cdl-a-doppler",
    "dft-4ue-cdl-a-lmmse",
    "dft-4ue-cdl-a-estimator-matrix",
    "dft-4ue-cdl-a-detector-sweep-gpu",
    "dft-4ue-cdl-a-detector-sweep-cpu",
    "dft-1ue-1tx-cdl-a",
    "dft-1ue-4layer-cdl-a",
    "dft-4ue-4tx-codebook-cdl-a",
    "dft-8stream-cdl-a",
    "dft-8stream-cdl-b",
    "dft-8stream-cdl-c",
    "dft-8stream-mmse-pic",
    "cp-2ue-2layer-cdl-a",
    "cp-2ue-2layer-cdl-b",
    "cp-2ue-2layer-cdl-c",
    "cp-2ue-2layer-cdl-d",
    "cp-2ue-2layer-cdl-e",
    "cp-2ue-2layer-type2-dmrs",
    "cp-2ue-2layer-cdl-a-lmmse",
    "cp-4ue-codebook-cp12",
)


def _config_sandbox(root: Path) -> Path:
    """Mirror the shipped config directory and publish synthetic accepted priors.

    Shipped ``configs/tap_power_prior/*.npz`` are ignored local artifacts, so a
    clean checkout has none and the DMRS-LMMSE bundles could not preflight.
    Configs are copied rather than symlinked because ``_profile_path`` resolves
    each profile and rejects anything outside the configured directory.
    """
    config_dir = root / "configs"
    config_dir.mkdir()
    for path in sorted((ROOT / "configs").glob("*.toml")):
        shutil.copyfile(path, config_dir / path.name)
    prior_dir = config_dir / "tap_power_prior"
    scratch = root / "scratch"
    scratch.mkdir()
    for tx_name, channel_name, simulation_name in (
        ("pusch_4ue.toml", "cdl_38_901_4x4.toml", "bler_dft_4ue_dmrs_lmmse_smoke.toml"),
        (
            "pusch_cp_2ue_2layer.toml",
            "cdl_38_901_2tx_4rx.toml",
            "bler_cp_2ue_2layer_dmrs_lmmse_smoke.toml",
        ),
    ):
        source = ROOT / "configs"
        tx_settings = TxSettings.from_toml(source / tx_name)
        channel_settings = ChannelSettings.from_toml(source / channel_name)
        simulation = BlerSettings.from_toml(source / simulation_name)
        transmitter = NrPuschTx(tx_settings, device="cpu")
        compatibility = dmrs_prior_compatibility(
            tx_settings,
            channel_settings,
            l_min=simulation.l_min,
            max_delay_spread_s=(
                simulation.max_delay_spread_s
                or channel_settings.channel.max_delay_spread_s
            ),
            fft_size=transmitter._tx_freq.resource_grid.fft_size,
            sample_rate_hz=transmitter.sample_rate_hz,
        )
        candidate = scratch / f"{tx_name}.prior.npz"
        save_dmrs_tap_power_prior(
            candidate,
            np.ones(compatibility["l_max"] - compatibility["l_min"] + 1),
            compatibility=compatibility,
            training_seed=7,
            training_realizations=32,
        )
        publish_accepted_dmrs_prior(
            prior_dir,
            candidate,
            compatibility=compatibility,
            acceptance_gate={"passed": True},
        )
    return config_dir


class WebEstimatorMatrixTest(unittest.TestCase):
    def test_web_lists_matrix_profile_and_counts_all_estimator_detector_snr_points(self):
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-matrix-") as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            config_dir.mkdir()
            for path in (TX_CONFIG, CHANNEL_CONFIG, SIMULATION_CONFIG):
                shutil.copyfile(path, config_dir / path.name)
            tx_settings = TxSettings.from_toml(TX_CONFIG)
            channel_settings = ChannelSettings.from_toml(CHANNEL_CONFIG)
            original_simulation = BlerSettings.from_toml(SIMULATION_CONFIG)
            transmitter = NrPuschTx(tx_settings, device="cpu")
            max_delay_spread_s = (
                original_simulation.max_delay_spread_s
                or channel_settings.channel.max_delay_spread_s
            )
            compatibility = dmrs_prior_compatibility(
                tx_settings,
                channel_settings,
                l_min=original_simulation.l_min,
                max_delay_spread_s=max_delay_spread_s,
                fft_size=transmitter._tx_freq.resource_grid.fft_size,
                sample_rate_hz=transmitter.sample_rate_hz,
            )
            candidate = root / "matrix.prior.npz"
            save_dmrs_tap_power_prior(
                candidate,
                np.ones(compatibility["l_max"] - compatibility["l_min"] + 1),
                compatibility=compatibility,
                training_seed=7,
                training_realizations=32,
            )
            published = publish_accepted_dmrs_prior(
                config_dir / "tap_power_prior",
                candidate,
                compatibility=compatibility,
                acceptance_gate={"passed": True},
            )
            self.assertEqual(published, prior_registry_path(config_dir / "tap_power_prior", compatibility))

            app = SimulationWebApp(config_dir, root / "runs")
            worker_entered = Event()
            app._execute = lambda _job_id: worker_entered.set()
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(f"{base_url}/api/configs", timeout=5) as response:
                    configs = json.loads(response.read())
                self.assertIn(SIMULATION_CONFIG.name, configs["simulation"])

                request_data = {}
                for kind, path in (
                    ("tx", TX_CONFIG),
                    ("channel", CHANNEL_CONFIG),
                    ("simulation", SIMULATION_CONFIG),
                ):
                    request_data[f"{kind}_name"] = path.name
                    request_data[f"{kind}_text"] = app.get_config(kind, path.name)["text"]
                preflight_request = Request(
                    f"{base_url}/api/validate-run",
                    data=json.dumps(request_data).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(preflight_request, timeout=5) as response:
                    preflight = json.loads(response.read())
                self.assertTrue(preflight["valid"], preflight["errors"])

                request = Request(
                    f"{base_url}/api/runs",
                    data=json.dumps(request_data).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=5) as response:
                    job = json.loads(response.read())
                simulation_settings = BlerSettings.from_toml(SIMULATION_CONFIG)
                self.assertEqual(
                    job["total_points"],
                    len(simulation_settings.channel_estimators_for_sweep)
                    * len(simulation_settings.detectors)
                    * len(simulation_settings.snr_db),
                )
                snapshot = (root / "runs" / job["id"] / "simulation.toml").read_text(encoding="utf-8")
                self.assertNotIn("dmrs_tap_power_prior_path", snapshot)
                BlerSettings.from_toml(root / "runs" / job["id"] / "simulation.toml")
                incompatible = dict(request_data)
                incompatible["simulation_text"] = request_data["simulation_text"].replace(
                    "l_min = -6", "l_min = -5", 1
                )
                preflight = app.validate_run(incompatible)
                self.assertFalse(preflight["valid"])
                self.assertTrue(
                    any("未找到" in error["message"] for error in preflight["errors"]),
                    preflight["errors"],
                )
                self.assertTrue(worker_entered.wait(5))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                app.shutdown()


    def test_web_saves_and_queues_tx_with_additional_dmrs_position_one(self):
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-additional-dmrs-") as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            config_dir.mkdir()
            simulation_config = ROOT / "configs" / "bler_smoke.toml"
            for path in (TX_CONFIG, CHANNEL_CONFIG, simulation_config):
                shutil.copyfile(path, config_dir / path.name)

            app = SimulationWebApp(config_dir, root / "runs")
            worker_entered = Event()
            app._execute = lambda _job_id: worker_entered.set()
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                tx_text = app.get_config("tx", TX_CONFIG.name)["text"]
                tx_text = tx_text.replace(
                    "dmrs_additional_position = 0",
                    "dmrs_additional_position = 1",
                    1,
                )
                save_request = Request(
                    f"{base_url}/api/configs/tx/{TX_CONFIG.name}",
                    data=json.dumps({"text": tx_text}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="PUT",
                )
                with urlopen(save_request, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    saved = json.loads(response.read())
                self.assertIn("dmrs_additional_position = 1", saved["text"])

                request_data = {
                    "tx_name": TX_CONFIG.name,
                    "tx_text": saved["text"],
                    "channel_name": CHANNEL_CONFIG.name,
                    "channel_text": app.get_config("channel", CHANNEL_CONFIG.name)["text"],
                    "simulation_name": simulation_config.name,
                    "simulation_text": app.get_config("simulation", simulation_config.name)["text"],
                }
                run_request = Request(
                    f"{base_url}/api/runs",
                    data=json.dumps(request_data).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(run_request, timeout=5) as response:
                    self.assertEqual(response.status, 201)
                    job = json.loads(response.read())
                self.assertTrue(worker_entered.wait(5))
                self.assertEqual(job["status"], "queued")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                app.shutdown()

    def test_web_saves_mimo_profile_through_common_validator(self):
        source = ROOT / "configs" / "pusch_2ue_4layer.toml"
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-mimo-") as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            config_dir.mkdir()
            shutil.copyfile(source, config_dir / source.name)
            app = SimulationWebApp(config_dir, root / "runs")
            try:
                text = source.read_text(encoding="utf-8")
                saved = app.save_config("tx", source.name, text)
                self.assertIn("dmrs_ports = [4, 5, 6, 7]", saved["text"])
                self.assertEqual(len(TxSettings.from_toml(config_dir / source.name).users), 2)
                with self.assertRaises(ValueError):
                    app.save_config(
                        "tx", source.name,
                        text.replace("dmrs_ports = [4, 5, 6, 7]",
                                     "dmrs_ports = [0, 5, 6, 7]"))
                self.assertEqual((config_dir / source.name).read_text(encoding="utf-8"), text)
            finally:
                app.shutdown()


    def test_curated_scenarios_preflight_as_complete_bundles(self):
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-scenarios-") as temporary:
            root = Path(temporary)
            config_dir = _config_sandbox(root)
            app = SimulationWebApp(config_dir, root / "runs")
            try:
                scenarios = app.list_scenarios()
                self.assertEqual([item["id"] for item in scenarios], list(SCENARIO_IDS))
                self.assertEqual(len(list((config_dir / "tap_power_prior").glob("*/*.npz"))), 2)
                for scenario in scenarios:
                    payload = {"scenario_id": scenario["id"]}
                    for kind, name in scenario["profiles"].items():
                        payload[f"{kind}_name"] = name
                        payload[f"{kind}_text"] = app.get_config(kind, name)["text"]
                    result = app.validate_run(payload)
                    self.assertTrue(result["valid"], (scenario["id"], result["errors"]))
                    self.assertEqual(result["scenario_id"], scenario["id"])
            finally:
                app.shutdown()

    def test_invalid_channel_pair_is_filtered_and_rejected_before_queueing(self):
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-compatibility-") as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            config_dir.mkdir()
            for path in (
                TX_CONFIG,
                CHANNEL_CONFIG,
                ROOT / "configs" / "cdl_38_901_2tx_4rx.toml",
                ROOT / "configs" / "bler_smoke.toml",
            ):
                shutil.copyfile(path, config_dir / path.name)
            app = SimulationWebApp(config_dir, root / "runs")
            try:
                payload = {}
                for kind, path in (
                    ("tx", TX_CONFIG),
                    ("channel", ROOT / "configs" / "cdl_38_901_2tx_4rx.toml"),
                    ("simulation", ROOT / "configs" / "bler_smoke.toml"),
                ):
                    payload[f"{kind}_name"] = path.name
                    payload[f"{kind}_text"] = app.get_config(kind, path.name)["text"]
                result = app.validate_run(payload)
                self.assertFalse(result["valid"])
                self.assertIn("必须等于 PUSCH", result["errors"][0]["message"])

                options = app.compatible_profiles("channel", payload)["profiles"]
                self.assertTrue(options[CHANNEL_CONFIG.name]["compatible"])
                self.assertFalse(options["cdl_38_901_2tx_4rx.toml"]["compatible"])
                with self.assertRaises(ApiError):
                    app.create_run(payload)
                self.assertEqual(list((root / "runs").iterdir()), [])
            finally:
                app.shutdown()

    def test_mcs_index_change_keeps_cp_dmrs_prior_compatible(self):
        tx_profile = ROOT / "configs" / "pusch_cp_2ue_2layer.toml"
        channel_profile = ROOT / "configs" / "cdl_38_901_2tx_4rx.toml"
        simulation_profile = ROOT / "configs" / "bler_cp_smoke.toml"
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-mcs-prior-") as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            config_dir.mkdir()
            for path in (tx_profile, channel_profile, simulation_profile):
                shutil.copyfile(path, config_dir / path.name)
            tx_settings = TxSettings.from_toml(tx_profile)
            channel_settings = ChannelSettings.from_toml(channel_profile)
            transmitter = NrPuschTx(tx_settings, device="cpu")
            simulation_settings = BlerSettings.from_toml(simulation_profile)
            l_min = simulation_settings.l_min
            max_delay_spread_s = (
                simulation_settings.max_delay_spread_s
                or channel_settings.channel.max_delay_spread_s
            )
            compatibility = dmrs_prior_compatibility(
                tx_settings,
                channel_settings,
                l_min=l_min,
                max_delay_spread_s=max_delay_spread_s,
                fft_size=transmitter._tx_freq.resource_grid.fft_size,
                sample_rate_hz=transmitter.sample_rate_hz,
            )
            candidate = config_dir / "cp-mcs.prior.npz"
            tap_count = compatibility["l_max"] - l_min + 1
            save_dmrs_tap_power_prior(
                candidate,
                np.ones(tap_count),
                compatibility=compatibility,
                training_seed=7,
                training_realizations=32,
            )
            publish_accepted_dmrs_prior(
                config_dir / "tap_power_prior",
                candidate,
                compatibility=compatibility,
                acceptance_gate={"passed": True},
            )
            simulation_text = simulation_profile.read_text(encoding="utf-8").replace(
                'channel_estimator = "dmrs"',
                'channel_estimator = "dmrs-lmmse"',
                1,
            )
            app = SimulationWebApp(config_dir, root / "runs")
            try:
                payload = {
                    "tx_name": tx_profile.name,
                    "tx_text": app.get_config("tx", tx_profile.name)["text"],
                    "channel_name": channel_profile.name,
                    "channel_text": app.get_config("channel", channel_profile.name)["text"],
                    "simulation_name": simulation_profile.name,
                    "simulation_text": simulation_text,
                }
                initial = app.validate_run(payload)
                self.assertTrue(initial["valid"], initial["errors"])

                edited = dict(payload)
                edited["tx_text"] = payload["tx_text"].replace(
                    "mcs_index = 8", "mcs_index = 20", 1
                )
                result = app.validate_run(edited)
                self.assertTrue(result["valid"], result["errors"])

                shutil.rmtree(app.prior_dir)
                missing = app.validate_run(payload)
                self.assertFalse(missing["valid"])
                self.assertTrue(
                    any("未找到" in error["message"] for error in missing["errors"]),
                    missing["errors"],
                )
            finally:
                app.shutdown()


if __name__ == "__main__":
    unittest.main()
