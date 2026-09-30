"""Web API integration test for uploaded external receive captures."""

import base64
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import shutil
import tempfile
from threading import Thread
from urllib.request import Request, urlopen
import unittest

import numpy as np

from nr_pusch.iq import read_matlab_rx_reference
from nr_pusch.web import SimulationWebApp, make_handler


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs" / "rx_pusch_4ue.toml"
FIXTURE = ROOT / "tests" / "fixtures" / "matlab_h5" / "RxTestVector.h5"
HIGH_RATE_CONFIG = ROOT / "configs" / "rx_pusch_4ue_mcs27.toml"
HIGH_RATE_FIXTURE = ROOT / "tests" / "fixtures" / "matlab_h5" / "RxTestVectorCase123427.h5"
SCRAMBLING_FIXTURE = ROOT / "tests" / "fixtures" / "matlab_h5" / "scrambSeqCase123427.h5"


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
                self.assertIn('id="rx-constellation-grid"', page)
                self.assertIn('id="rx-scrambling"', page)
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
                    "noise_variance": 0.003,
                    "detector": "mmse-pic",
                    "detector_parameter": 16,
                    "detector_damping": 0.75,
                    "max_delay_spread_s": 3e-6,
                    "device": "cpu",
                }).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read())
                self.assertEqual(payload["crc_status"], [True, True, True, True])
                self.assertEqual(payload["crc_labels"], ["ue0", "ue1", "ue2", "ue3"])
                self.assertEqual(payload["crc_pass_count"], 4)
                self.assertEqual(payload["block_count"], 4)
                self.assertTrue(payload["reference_comparison"]["exact_match"])
                self.assertEqual(payload["reference_comparison"]["bit_errors"], [0, 0, 0, 0])
                self.assertEqual(payload["input"]["format"], "matlab-h5")
                self.assertEqual([user["name"] for user in payload["constellation"]["users"]], ["ue0", "ue1", "ue2", "ue3"])
                self.assertTrue(payload["constellation"]["users"][0]["real"])
                self.assertEqual(
                    len(payload["constellation"]["users"][0]["real"]),
                    len(payload["constellation"]["users"][0]["imag"]),
                )
                self.assertTrue(payload["cb_crc"]["available"])
                self.assertEqual(
                    [user["name"] for user in payload["cb_crc"]["users"]],
                    ["ue0", "ue1", "ue2", "ue3"],
                )
                for user in payload["cb_crc"]["users"]:
                    self.assertEqual(user["total"], 4)
                    self.assertEqual(user["passed"], 4)
                    self.assertEqual(user["status"], [True] * 4)
                for output_url in (payload["output_npz_url"], payload["output_json_url"]):
                    with urlopen(f"{base_url}{output_url}", timeout=5) as response:
                        self.assertGreater(len(response.read()), 0)
                with urlopen(f"{base_url}{payload['output_npz_url']}", timeout=5) as response:
                    with np.load(io.BytesIO(response.read()), allow_pickle=False) as decoded:
                        decoded_bits = decoded["bits"][0]
                np.testing.assert_array_equal(
                    decoded_bits, read_matlab_rx_reference(FIXTURE).transmitted_bits
                )

                default_request = Request(f"{base_url}/api/rx/decode", data=json.dumps({
                    "use_default_capture": True,
                    "device": "cpu",
                }).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(default_request, timeout=60) as response:
                    default_result = json.loads(response.read())
                self.assertTrue(default_result["used_default_capture"])
                self.assertEqual(default_result["input"]["source"], "repository-default")
                self.assertEqual(default_result["crc_status"], [True, True, True, True])
                self.assertTrue(default_result["reference_comparison"]["exact_match"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                app.shutdown()

    def test_soft_mmse_pic_profile_is_used_without_request_overrides(self):
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-soft-rx-") as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            config_dir.mkdir()
            app = SimulationWebApp(config_dir, root / "runs")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            soft_profile = CONFIG.read_text(encoding="utf-8")
            soft_profile = soft_profile.replace(
                'detector = "mmse-pic"', 'detector = "soft-mmse-pic"'
            ).replace("detector_parameter = 8\n", "").replace(
                "detector_damping = 0.5\n", ""
            )
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/rx/decode",
                    data=json.dumps({
                        "config_name": "",
                        "config_text": soft_profile,
                        "input_name": FIXTURE.name,
                        "input_format": "matlab-h5",
                        "input_domain": "frequency",
                        "input_base64": base64.b64encode(
                            FIXTURE.read_bytes()
                        ).decode("ascii"),
                        "device": "cpu",
                    }).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                app.shutdown()

        self.assertEqual(payload["detector"], "soft-mmse-pic")
        self.assertEqual(payload["receiver"]["detector_parameter"], 1)
        self.assertEqual(payload["receiver"]["detector_damping"], 0.25)
        self.assertEqual(payload["receiver"]["detector_feedback_iterations"], 1)
        self.assertEqual(payload["crc_status"], [True, True, True, True])


    def test_high_rate_capture_reports_per_user_code_block_crc(self):
        """The profile's [receiver] table must drive the decode, and CB CRC must be per user."""
        with tempfile.TemporaryDirectory(prefix="nr-pusch-web-rx-mcs27-") as temporary:
            root = Path(temporary)
            config_dir = root / "configs"
            fixture_dir = root / "tests" / "fixtures" / "matlab_h5"
            config_dir.mkdir()
            fixture_dir.mkdir(parents=True)
            shutil.copyfile(HIGH_RATE_CONFIG, config_dir / HIGH_RATE_CONFIG.name)
            shutil.copyfile(HIGH_RATE_FIXTURE, fixture_dir / HIGH_RATE_FIXTURE.name)
            shutil.copyfile(SCRAMBLING_FIXTURE, fixture_dir / SCRAMBLING_FIXTURE.name)
            app = SimulationWebApp(config_dir, root / "runs")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                # No receiver knobs in the request: every one must come from the profile.
                request = Request(f"{base_url}/api/rx/decode", data=json.dumps({
                    "config_name": HIGH_RATE_CONFIG.name,
                    "input_name": HIGH_RATE_FIXTURE.name,
                    "input_format": "matlab-h5",
                    "input_base64": base64.b64encode(
                        HIGH_RATE_FIXTURE.read_bytes()
                    ).decode("ascii"),
                    "scrambling_name": SCRAMBLING_FIXTURE.name,
                    "scrambling_base64": base64.b64encode(
                        SCRAMBLING_FIXTURE.read_bytes()
                    ).decode("ascii"),
                    "device": "cpu",
                }).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=300) as response:
                    payload = json.loads(response.read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                app.shutdown()
        self.assertEqual(payload["input"]["scrambling"], SCRAMBLING_FIXTURE.name)
        self.assertEqual(payload["crc_status"], [True, True, True, False])
        self.assertTrue(payload["cb_crc"]["available"])
        self.assertEqual(
            [user["name"] for user in payload["cb_crc"]["users"]],
            ["ue0", "ue1", "ue2", "ue3"],
        )
        # max_delay_spread_s = 2e-6 from the profile keeps ue2 at six of six blocks;
        # the web app's own 3e-6 fallback would drop it to five.
        self.assertEqual([user["passed"] for user in payload["cb_crc"]["users"]], [6, 6, 6, 0])
        self.assertEqual(payload["cb_crc"]["users"][3]["total"], 6)
        self.assertEqual(payload["cb_crc"]["users"][3]["status"], [False] * 6)


if __name__ == "__main__":
    unittest.main()
