"""Web API coverage for estimator-detector matrix runs."""

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
from threading import Event, Thread
from urllib.request import Request, urlopen
import unittest

from nr_pusch.simulation_config import BlerSettings
from nr_pusch.web import SimulationWebApp, make_handler


ROOT = Path(__file__).parents[2]
TX_CONFIG = ROOT / "configs" / "pusch_4ue.toml"
CHANNEL_CONFIG = ROOT / "configs" / "cdl_38_901_4x4.toml"
SIMULATION_CONFIG = ROOT / "configs" / "bler_estimator_matrix.toml"


class WebEstimatorMatrixTest(unittest.TestCase):
    def test_web_lists_matrix_profile_and_counts_all_estimator_detector_snr_points(self):
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-matrix-") as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            config_dir.mkdir()
            for path in (TX_CONFIG, CHANNEL_CONFIG, SIMULATION_CONFIG):
                shutil.copyfile(path, config_dir / path.name)

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
                self.assertTrue(worker_entered.wait(5))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                app.shutdown()


if __name__ == "__main__":
    unittest.main()
