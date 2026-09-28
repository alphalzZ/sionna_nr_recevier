"""Web API integration test for uploaded external receive captures."""

import base64
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
from threading import Thread
from urllib.request import Request, urlopen
import unittest

from nr_pusch.web import SimulationWebApp, make_handler


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs" / "rx_pusch_4ue.toml"
FIXTURE = ROOT / "tests" / "fixtures" / "matlab_h5" / "RxTestVector.h5"


class WebRxImportTest(unittest.TestCase):
    def test_uploaded_matlab_h5_returns_crc_constellation_and_downloads(self):
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-rx-") as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            fixture_dir = root / "tests" / "fixtures" / "matlab_h5"
            runs_dir = root / "runs"
            config_dir.mkdir()
            fixture_dir.mkdir(parents=True)
            shutil.copyfile(CONFIG, config_dir / CONFIG.name)
            shutil.copyfile(FIXTURE, fixture_dir / FIXTURE.name)
            app = SimulationWebApp(config_dir, runs_dir)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base_url, timeout=5) as response:
                    page = response.read().decode("utf-8")
                self.assertIn('id="rx-workflow-tab"', page)
                self.assertIn('id="rx-constellation"', page)
                with urlopen(f"{base_url}/api/rx/configs", timeout=5) as response:
                    listed = json.loads(response.read())
                self.assertEqual(listed["configs"], [CONFIG.name])
                with urlopen(f"{base_url}/api/rx/defaults", timeout=5) as response:
                    defaults = json.loads(response.read())
                self.assertTrue(defaults["available"])
                self.assertEqual(defaults["config_name"], CONFIG.name)
                self.assertEqual(defaults["input_name"], FIXTURE.name)
                request = Request(f"{base_url}/api/rx/decode", data=json.dumps({
                    "config_name": "",
                    "config_text": CONFIG.read_text(encoding="utf-8"),
                    "input_name": FIXTURE.name,
                    "input_format": "matlab-h5",
                    "input_domain": "frequency",
                    "input_base64": base64.b64encode(FIXTURE.read_bytes()).decode("ascii"),
                    "noise_variance": 0.0,
                    "detector": "lmmse",
                    "device": "cpu",
                }).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read())
                self.assertEqual(payload["crc_status"], [True, True, True, True])
                self.assertEqual(payload["crc_labels"], ["ue0", "ue1", "ue2", "ue3"])
                self.assertEqual(payload["crc_pass_count"], 4)
                self.assertEqual(payload["block_count"], 4)
                self.assertEqual(payload["input"]["format"], "matlab-h5")
                self.assertEqual([user["name"] for user in payload["constellation"]["users"]], ["ue0", "ue1", "ue2", "ue3"])
                self.assertTrue(payload["constellation"]["users"][0]["real"])
                self.assertEqual(
                    len(payload["constellation"]["users"][0]["real"]),
                    len(payload["constellation"]["users"][0]["imag"]),
                )
                for output_url in (payload["output_npz_url"], payload["output_json_url"]):
                    with urlopen(f"{base_url}{output_url}", timeout=5) as response:
                        self.assertGreater(len(response.read()), 0)

                default_request = Request(f"{base_url}/api/rx/decode", data=json.dumps({
                    "use_default_capture": True,
                    "noise_variance": 0.0,
                    "detector": "lmmse",
                    "device": "cpu",
                }).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(default_request, timeout=60) as response:
                    default_result = json.loads(response.read())
                self.assertTrue(default_result["used_default_capture"])
                self.assertEqual(default_result["input"]["source"], "repository-default")
                self.assertEqual(default_result["crc_status"], [True, True, True, True])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                app.shutdown()


if __name__ == "__main__":
    unittest.main()
