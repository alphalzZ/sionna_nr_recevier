from __future__ import annotations

import base64
from csv import DictReader, DictWriter
from io import BytesIO, StringIO
import json
from pathlib import Path
import subprocess
import random
import shutil
import string
import struct
import tempfile
import threading
import time
from unittest.mock import patch
import zipfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import unittest

import numpy as np
from PIL import Image

from nr_pusch.rt_config import RtBeamSettings
from nr_pusch.rt_scene_assets import copy_scene_assets, resolve_builtin_scene_assets
from nr_pusch.web import ApiError, SimulationWebApp, make_handler


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
SCENARIOS = {
    "rt-los-quick": ("pusch_4ue.toml", "rt_beam_los.toml"),
    "rt-ground-wall-quick": ("pusch_4ue.toml", "rt_beam_ground_wall.toml"),
    "rt-cp-los-quick": ("pusch_rt_4ue_cp.toml", "rt_beam_los.toml"),
}

_RT_TEST_QUICK_PROFILE = """[bler]
snr_db = [20.0, 40.0, 60.0]
batch_size = 1
max_frames_per_snr = 2
target_block_errors = 9
seed = 13
num_decoder_iterations = 20
channel_estimator = "dmrs"
channel_estimators = ["dmrs"]
detector = "lmmse"
detectors = ["beam-independent", "zf", "lmmse"]
device = "cpu"
channel_domain = "frequency"
l_min = -6
max_delay_spread_s = 3e-6
stop_at_zero_bler = false
"""


class WebRtBlerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nr-pusch-web-rt-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        for path in CONFIGS.glob("*.toml"):
            shutil.copyfile(path, self.config_dir / path.name)
        (self.config_dir / "bler_rt_beam_web_quick.toml").write_text(
            _RT_TEST_QUICK_PROFILE, encoding="utf-8"
        )
        shutil.copyfile(CONFIGS / "scenarios.toml", self.config_dir / "scenarios.toml")
        self.runs_dir = self.root / "runs"
        self.app = SimulationWebApp(self.config_dir, self.runs_dir)
        self.outputs_dir = self.root / "outputs"
        self.app.outputs_dir = self.outputs_dir
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=5)
        self.app.shutdown()

    def _request(self, path: str, payload: dict | None = None, *, method: str = "GET"):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method,
        )
        with urlopen(request, timeout=30) as response:
            return response.status, response.headers, response.read()

    def _json(self, path: str, payload: dict | None = None, *, method: str = "GET") -> dict:
        status, _, body = self._request(path, payload, method=method)
        self.assertIn(status, {200, 201})
        return json.loads(body)

    def test_rt_options_report_long_profile_limits(self):
        options = self._json("/api/rt/options")
        self.assertEqual(options["web_limits"]["batch_size"], [1, 20])
        self.assertEqual(options["web_limits"]["max_frames_per_snr"], [1, 2_000])
        self.assertEqual(options["web_limits"]["stop_at_zero_bler"], [False, True])
        self.assertEqual(options["web_limits"]["array_rows"], [1, 16])
        self.assertEqual(options["web_limits"]["array_cols"], [1, 16])
        self.assertEqual(options["web_limits"]["array_elements"], [4, 256])
        self.assertEqual(
            {item["id"] for item in options["scene_presets"]},
            {
                "empty", "ground", "ground_wall", "box", "box_knife",
                "box_one_screen", "box_two_screens", "double_reflector",
                "etoile", "floor_wall", "florence", "munich", "san_francisco",
                "simple_reflector", "simple_street_canyon",
                "simple_street_canyon_with_cars", "simple_wedge", "triple_reflector",
            },
        )

    def test_web_preflight_accepts_maximum_16_by_16_receiver_array(self):
        settings = RtBeamSettings.from_toml(CONFIGS / "rt_beam_los.toml").to_dict()
        settings["receiver"]["num_rows"] = 16
        settings["receiver"]["num_cols"] = 16
        accepted = self._json("/api/rt/config", {"rt_settings": settings}, method="POST")
        self.assertEqual(
            accepted["summary"]["receiver_array"],
            {"rows": 16, "cols": 16, "elements": 256},
        )

        oversized = RtBeamSettings.from_toml(CONFIGS / "rt_beam_los.toml").to_dict()
        oversized["receiver"]["num_rows"] = 17
        with self.assertRaises(HTTPError) as rejected:
            self._request("/api/rt/config", {"rt_settings": oversized}, method="POST")
        self.assertEqual(rejected.exception.code, 400)
        rejected.exception.close()

    def test_scene_preview_renders_builtin_geometry_and_devices_as_png(self):
        settings = RtBeamSettings.from_toml(CONFIGS / "rt_beam_los.toml").to_dict()
        settings["rt"]["scene"] = "simple_street_canyon"
        with self.assertRaises(ApiError):
            self.app.rt_scene_preview({"rt_settings": settings, "view": []})
        status, headers, image = self._request(
            "/api/rt/scene-preview",
            {"rt_settings": settings, "view": "oblique"},
            method="POST",
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "image/png")
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack_from(">II", image, 16), (640, 400))
        pixels = np.asarray(Image.open(BytesIO(image)).convert("RGBA"))
        self.assertGreater(np.count_nonzero(pixels[:, :, 3]), 1_000)
        rgb = pixels[:, :, :3].astype(np.float32)
        ue_color_counts = [
            np.count_nonzero((rgb[:, :, 2] > 1.5 * rgb[:, :, 0]) & (rgb[:, :, 2] > 1.2 * rgb[:, :, 1])),
            np.count_nonzero((rgb[:, :, 0] > 1.2 * rgb[:, :, 1]) & (rgb[:, :, 1] > 1.2 * rgb[:, :, 2])),
            np.count_nonzero((rgb[:, :, 2] > 1.1 * rgb[:, :, 0]) & (rgb[:, :, 0] > 1.15 * rgb[:, :, 1])),
            np.count_nonzero((rgb[:, :, 1] > 1.2 * rgb[:, :, 0]) & (rgb[:, :, 1] > 1.1 * rgb[:, :, 2])),
        ]
        self.assertTrue(all(count >= 5 for count in ue_color_counts))
        top_status, _, top_image = self._request(
            "/api/rt/scene-preview",
            {"rt_settings": settings, "view": "top"},
            method="POST",
        )
        self.assertEqual(top_status, 200)
        self.assertNotEqual(top_image, image)

    def test_osm_scene_zip_upload_renders_geometry_and_devices(self):
        archive_path = ROOT / "blender_scene" / "test_scene" / "sionna_rt_export.zip"
        upload = self._json(
            "/api/rt/scenes",
            {
                "filename": archive_path.name,
                "content_base64": base64.b64encode(archive_path.read_bytes()).decode("ascii"),
            },
            method="POST",
        )
        self.assertEqual(upload["mesh_summary"]["meshes/buildings.ply"]["vertex_count"], 3348)
        self.assertEqual(upload["mesh_summary"]["meshes/buildings.ply"]["face_count"], 4768)
        self.assertEqual(upload["mesh_summary"]["meshes/ground.ply"]["vertex_count"], 4)

        settings = RtBeamSettings.from_toml(CONFIGS / "rt_beam_los.toml").to_dict()
        settings["rt"]["scene"] = "custom"
        settings["rt"]["scene_file"] = "scene.xml"
        images = []
        for view in ("oblique", "top"):
            status, headers, image = self._request(
                "/api/rt/scene-preview",
                {"rt_settings": settings, "scene_id": upload["scene_id"], "view": view},
                method="POST",
            )
            self.assertEqual(status, 200)
            self.assertEqual(headers.get_content_type(), "image/png")
            self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(struct.unpack_from(">II", image, 16), (640, 400))
            pixels = np.asarray(Image.open(BytesIO(image)).convert("RGBA"))
            self.assertGreater(np.count_nonzero(pixels[:, :, 3]), 10_000)
            images.append(image)
            if view == "oblique":
                rgb = pixels[:, :, :3].astype(np.float32)
                ue_color_counts = [
                    np.count_nonzero((rgb[:, :, 2] > 1.5 * rgb[:, :, 0]) & (rgb[:, :, 2] > 1.2 * rgb[:, :, 1])),
                    np.count_nonzero((rgb[:, :, 0] > 1.2 * rgb[:, :, 1]) & (rgb[:, :, 1] > 1.2 * rgb[:, :, 2])),
                    np.count_nonzero((rgb[:, :, 2] > 1.1 * rgb[:, :, 0]) & (rgb[:, :, 0] > 1.15 * rgb[:, :, 1])),
                    np.count_nonzero((rgb[:, :, 1] > 1.2 * rgb[:, :, 0]) & (rgb[:, :, 1] > 1.1 * rgb[:, :, 2])),
                ]
                self.assertTrue(all(count >= 5 for count in ue_color_counts))
        self.assertNotEqual(images[0], images[1])

    def _seed_history_job(self, job_id: str, status: str) -> Path:
        directory = self.runs_dir / job_id
        directory.mkdir()
        job = {
            "id": job_id,
            "status": status,
            "created_at": "2026-10-05T00:00:00+00:00",
            "channel_backend": "rt",
            "config_names": {"tx": "pusch_4ue.toml", "rt": "rt_beam_los.toml"},
        }
        (directory / "job.json").write_text(json.dumps(job), encoding="utf-8")
        self.app._jobs[job_id] = job
        return directory

    def test_delete_history_single_and_batch_removes_job_archives_only(self):
        run_ids = ("a1b2c3d4e5f6", "b2c3d4e5f6a1", "c3d4e5f6a1b2")
        for job_id in run_ids:
            self._seed_history_job(job_id, "completed")
            archive = self.outputs_dir / "rt_runs" / job_id
            archive.mkdir(parents=True)
            (archive / "results.csv").write_text("snr_db,bler\n", encoding="utf-8")
        shared_cache = self.outputs_dir / "rt_snapshots" / "shared-key"
        shared_cache.mkdir(parents=True)
        (shared_cache / "cache.json").write_text("{}", encoding="utf-8")

        single = self._json(f"/api/runs/{run_ids[0]}", method="DELETE")
        self.assertEqual(single, {"deleted_ids": [run_ids[0]], "deleted_count": 1})
        self.assertFalse((self.runs_dir / run_ids[0]).exists())
        self.assertFalse((self.outputs_dir / "rt_runs" / run_ids[0]).exists())

        batch_ids = list(run_ids[1:])
        batch = self._json("/api/runs", {"ids": batch_ids}, method="DELETE")
        self.assertEqual(batch, {"deleted_ids": batch_ids, "deleted_count": 2})
        for job_id in batch_ids:
            self.assertFalse((self.runs_dir / job_id).exists())
            self.assertFalse((self.outputs_dir / "rt_runs" / job_id).exists())
        self.assertTrue((shared_cache / "cache.json").is_file())
        self.assertEqual(self._json("/api/runs")["runs"], [])

        with self.assertRaises(HTTPError) as missing:
            self._request(f"/api/runs/{run_ids[0]}")
        self.assertEqual(missing.exception.code, 404)
        missing.exception.close()

    def test_delete_history_rejects_active_batch_and_skips_removed_queue_item(self):
        completed_id, running_id, queued_id, cancelled_id = (
            "c3d4e5f6a1b2", "d4e5f6a1b2c3", "f6a1b2c3d4e5", "e5f6a1b2c3d4",
        )
        completed_dir = self._seed_history_job(completed_id, "completed")
        running_dir = self._seed_history_job(running_id, "running")
        queued_dir = self._seed_history_job(queued_id, "queued")
        cancelled_dir = self._seed_history_job(cancelled_id, "cancelled")
        with self.assertRaises(HTTPError) as active:
            self._request(
                "/api/runs",
                {"ids": [completed_id, running_id, queued_id]},
                method="DELETE",
            )
        self.assertEqual(active.exception.code, 409)
        active.exception.close()
        self.assertTrue(completed_dir.is_dir())
        self.assertTrue(running_dir.is_dir())
        self.assertTrue(queued_dir.is_dir())
        self.assertIn(completed_id, self.app._jobs)
        self.assertIn(running_id, self.app._jobs)
        self.assertIn(queued_id, self.app._jobs)

        self._json(f"/api/runs/{cancelled_id}", method="DELETE")
        self.assertFalse(cancelled_dir.exists())
        self.app._execute(cancelled_id)
        self.assertNotIn(cancelled_id, self.app._jobs)


    def _scene_payload(self, scenario_id: str) -> dict[str, str]:
        listed = next(
            item for item in self._json("/api/scenarios")["scenarios"]
            if item["id"] == scenario_id
        )
        payload: dict[str, str] = {
            "channel_backend": "rt",
            "scenario_id": scenario_id,
        }
        for kind in ("tx", "rt", "simulation"):
            name = listed["profiles"][kind]
            profile = self._json(f"/api/configs/{kind}/{name}")
            payload[f"{kind}_name"] = name
            payload[f"{kind}_text"] = profile["text"]
        return payload

    def _submit_and_wait(self, scenario_id: str) -> dict:
        payload = self._scene_payload(scenario_id)
        preflight = self._json("/api/validate-run", payload, method="POST")
        self.assertTrue(preflight["valid"], preflight["errors"])
        self.assertEqual(preflight["total_points"], 9)
        job = self._json("/api/runs", payload, method="POST")
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            job = self._json(f"/api/runs/{job['id']}")
            if job["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.5)
        self.assertEqual(job["status"], "completed", (job.get("error"), job.get("logs")))
        self.assertEqual(job["stage"], "done")
        self.assertEqual(job["completed_points"], 9)
        self.assertEqual(len(job["points"]), 45)
        return job

    def test_scene_upload_http_limit_is_scoped_to_12_mib(self):
        connection = HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=30
        )
        try:
            connection.putrequest("POST", "/api/rt/scenes")
            connection.putheader("Host", f"127.0.0.1:{self.server.server_port}")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(12 * 1024 * 1024 + 1))
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 413)
            response.read()
        finally:
            connection.close()
        self.assertEqual(list(self.app.scene_store_dir.iterdir()), [])

    def test_real_los_ground_wall_and_cp_jobs_complete_and_replay(self):
        _, template_headers, template = self._request("/api/rt/scene-template.zip")
        self.assertEqual(template_headers.get_content_type(), "application/zip")
        with zipfile.ZipFile(BytesIO(template)) as source:
            files = {name: source.read(name) for name in source.namelist()}
        noise = "".join(random.Random(27).choices(string.ascii_letters + string.digits, k=600_000))
        mesh = files["meshes/ground.ply"]
        files["meshes/ground.ply"] = mesh.replace(
            b"format ascii 1.0\n",
            b"format ascii 1.0\ncomment " + noise.encode("ascii") + b"\n",
            1,
        )
        large_zip = BytesIO()
        with zipfile.ZipFile(large_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, contents in files.items():
                archive.writestr(name, contents)
        encoded = base64.b64encode(large_zip.getvalue()).decode("ascii")
        upload_body = json.dumps({"filename": "template.zip", "content_base64": encoded}).encode()
        self.assertGreater(len(upload_body), 512_000)
        request = Request(
            self.base + "/api/rt/scenes",
            data=upload_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            self.assertEqual(response.status, 201)
            uploaded = json.loads(response.read())
        self.assertEqual(len(uploaded["scene_id"]), 12)
        self.assertEqual(uploaded["mesh_summary"]["meshes/ground.ply"]["vertex_count"], 4)
        with self.assertRaises(HTTPError) as invalid:
            bad = Request(
                self.base + "/api/rt/scenes",
                data=json.dumps({"filename": "bad.zip", "content_base64": "bm90LXppcA=="}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urlopen(bad, timeout=10)
        self.assertEqual(invalid.exception.code, 400)
        invalid.exception.close()
        self.assertEqual(len(list(self.app.scene_store_dir.iterdir())), 1)

        completed = {}
        for scenario_id in SCENARIOS:
            job = self._submit_and_wait(scenario_id)
            completed[scenario_id] = job
            names = SCENARIOS[scenario_id]
            self.assertIs(job["snapshot_cache_hit"], False)
            self.assertRegex(job["snapshot_cache_key"], r"^[0-9a-f]{64}$")
            output_archive = self.outputs_dir / "rt_runs" / job["id"]
            self.assertEqual(job["output_archive"], f"outputs/rt_runs/{job['id']}")
            self.assertTrue((output_archive / "results.csv").is_file())
            self.assertTrue((output_archive / "channel_snapshot.npz").is_file())
            archived_job = json.loads((output_archive / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(archived_job["status"], "completed")
            cache_info = json.loads(
                (output_archive / "snapshot_cache.json").read_text(encoding="utf-8")
            )
            self.assertFalse(cache_info["hit"])
            self.assertEqual(job["config_names"]["tx"], names[0])
            self.assertEqual(job["config_names"]["rt"], names[1])
            aggregate = [row for row in job["points"] if row["user"] == "all"]
            self.assertEqual(len(aggregate), 9)
            self.assertTrue(all(row["transport_blocks"] == 8 for row in aggregate))
            self.assertTrue(all(row["frames"] == 2 for row in job["points"]))
            self.assertTrue(all(row["status"] in {"complete", "infeasible_rank", "singular_noise_covariance"} for row in job["points"]))
            self.assertEqual(len({(row["channel_estimator"], row["detector"], row["snr_db"], row["user"]) for row in job["points"]}), 45)

            report = self._json(f"/api/runs/{job['id']}/validation.json")
            self.assertTrue(report["passed"])
            self.assertEqual(report["policy"], "web-frequency-v1")
            strict_expected = scenario_id != "rt-ground-wall-quick"
            self.assertEqual(report["strict_fd_td_passed"], strict_expected)
            self.assertEqual(job["validation_summary"]["strict_fd_td_passed"], strict_expected)
            self.assertEqual(job["validation_summary"]["scene_bundle_sha256"], report["scene_bundle_sha256"])
            self.assertEqual(job["output_validation"], f"/api/runs/{job['id']}/validation.json")

            csv_status, csv_headers, csv_bytes = self._request(f"/api/runs/{job['id']}/results.csv")
            self.assertEqual(csv_status, 200)
            self.assertEqual(csv_headers.get_content_type(), "text/csv")
            csv_rows = list(DictReader(StringIO(csv_bytes.decode("utf-8"))))
            self.assertEqual(len(csv_rows), 45)
            manifest = self._json(f"/api/runs/{job['id']}/results.json")
            self.assertEqual(manifest["web_validation"]["policy"], "web-frequency-v1")
            self.assertEqual(manifest["scene_bundle_sha256"], report["scene_bundle_sha256"])
            self.assertEqual(manifest["channel_snapshot_scene_source"], "builtin")
            self.assertIn("noise_covariance_by_snr_db", manifest)
            self.assertEqual(manifest["rt_settings"]["rt"]["scene"], "ground_wall" if "wall" in scenario_id else "empty")

            snapshot_status, _, snapshot_bytes = self._request(f"/api/runs/{job['id']}/channel_snapshot.npz")
            self.assertEqual(snapshot_status, 200)
            self.assertTrue(snapshot_bytes.startswith(b"PK"))
            archive_status, archive_headers, archive_bytes = self._request(
                f"/api/runs/{job['id']}/reproducibility.zip"
            )
            self.assertEqual(archive_status, 200)
            self.assertEqual(archive_headers.get_content_type(), "application/zip")
            with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
                names = set(archive.namelist())
            self.assertTrue({"tx.toml", "rt.toml", "simulation.toml", "validation.json"}.issubset(names))
            self.assertTrue(any(name.startswith("scene/") for name in names))
            self.assertNotIn("run.log", names)
            self.assertNotIn("job.json", names)
            with self.assertRaises(ApiError):
                self.app.result_file(job["id"], "../job.json")

            stage_events = [
                json.loads(line)
                for line in (self.runs_dir / job["id"] / "stage.jsonl").read_text().splitlines()
            ]
            self.assertEqual([event["stage"] for event in stage_events], ["tracing", "validating"])
            self.assertTrue(all(event["type"] == "stage" and "at" in event for event in stage_events))

        first_los = completed["rt-los-quick"]
        cached_repeat = self._submit_and_wait("rt-los-quick")
        self.assertIs(cached_repeat["snapshot_cache_hit"], True)
        self.assertEqual(cached_repeat["snapshot_cache_key"], first_los["snapshot_cache_key"])
        self.assertNotEqual(cached_repeat["output_archive"], first_los["output_archive"])

        history = self._json("/api/runs")["runs"]
        for scenario_id, job in completed.items():
            saved = next(run for run in history if run["id"] == job["id"])
            self.assertEqual(
                saved["validation_summary"]["strict_fd_td_passed"],
                scenario_id != "rt-ground-wall-quick",
            )
    def test_rt_progress_counts_unique_complete_points_and_preserves_null_csv_values(self):
        user_names = ["ue0", "ue1", "ue2", "ue3"]
        rows = [
            {
                "channel_estimator": "dmrs",
                "detector": "lmmse",
                "device": "cpu",
                "snr_db": 20.0,
                "user": user,
                "frames": 2,
                "transport_blocks": 2 if user != "all" else 8,
                "block_errors": 0,
                "bler": 0.0,
                "actual_snr_db": None,
                "status": "complete",
                "reason": None,
            }
            for user in (*user_names, "all")
        ]
        progress_job_id = "c1d2e3f4a5b6"
        progress_dir = self.runs_dir / progress_job_id
        progress_dir.mkdir()
        progress = progress_dir / "progress.jsonl"
        event = {"type": "point", "rows": rows}
        progress.write_text(
            json.dumps(event) + "\n"
            + json.dumps(event) + "\n"
            + json.dumps({"type": "point", "rows": rows[:-1]}) + "\n"
            + '{"type":"point"',
            encoding="utf-8",
        )
        progress_job = {
            "id": progress_job_id,
            "status": "running",
            "created_at": "2026-10-04T00:00:00+00:00",
            "channel_backend": "rt",
            "user_names": user_names,
            "total_points": 1,
            "validation_summary": None,
        }
        with self.app._lock:
            self.app._jobs[progress_job_id] = progress_job
            self.app._write_job(progress_job)
        self.assertEqual(len(self.app._points(progress_job_id)), 5)
        self.assertEqual(self.app.get_run(progress_job_id)["completed_points"], 1)

        skipped_id = "c2d3e4f5a6b7"
        skipped_dir = self.runs_dir / skipped_id
        skipped_dir.mkdir()
        skipped_rows = [
            {
                **row,
                "snr_db": 40.0,
                "frames": 0,
                "transport_blocks": 0,
                "block_errors": 0,
                "bler": None,
                "bler_ci95_low": None,
                "bler_ci95_high": None,
                "crc_fail_rate": None,
                "ber": None,
                "actual_snr_db": None,
                "runtime_s": 0.0,
                "status": "skipped",
                "reason": "stop_at_zero_bler after zero errors at 20 dB",
            }
            for row in rows
        ]
        (skipped_dir / "progress.jsonl").write_text(
            json.dumps({"type": "point", "rows": skipped_rows}) + "\n",
            encoding="utf-8",
        )
        skipped_job = {
            "id": skipped_id,
            "status": "running",
            "created_at": "2026-10-04T00:00:02+00:00",
            "channel_backend": "rt",
            "user_names": user_names,
            "total_points": 1,
            "validation_summary": None,
        }
        with self.app._lock:
            self.app._jobs[skipped_id] = skipped_job
            self.app._write_job(skipped_job)
        projected_skips = self.app.get_run(skipped_id)
        self.assertEqual(projected_skips["completed_points"], 1)
        self.assertEqual(len(projected_skips["points"]), 5)
        self.assertTrue(all(row["status"] == "skipped" for row in projected_skips["points"]))
        self.assertTrue(all(row["bler"] is None for row in projected_skips["points"]))

        csv_job_id = "d1e2f3a4b5c6"
        csv_dir = self.runs_dir / csv_job_id
        csv_dir.mkdir()
        csv_path = csv_dir / "results.csv"
        csv_rows = [dict(row) for row in rows]
        csv_rows[1]["bler"] = ""
        csv_rows.append(dict(csv_rows[0]))
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        csv_job = {
            "id": csv_job_id,
            "status": "completed",
            "created_at": "2026-10-04T00:00:01+00:00",
            "channel_backend": "rt",
            "user_names": user_names,
            "total_points": 1,
            "validation_summary": None,
        }
        with self.app._lock:
            self.app._jobs[csv_job_id] = csv_job
            self.app._write_job(csv_job)
        parsed = self.app._points(csv_job_id)
        self.assertEqual(len(parsed), 5)
        self.assertEqual(parsed[1]["bler"], None)
        self.assertIsNone(parsed[0]["actual_snr_db"])
        self.assertEqual(self.app.get_run(csv_job_id)["completed_points"], 1)
    def test_queued_cancel_prepare_timeout_missing_report_and_restart_recovery(self):
        queued_id = "e1f2a3b4c5d6"
        self._write_manual_job(queued_id)
        queued = self.app.cancel_run(queued_id)
        self.assertEqual(queued["status"], "cancelled")
        with patch.object(self.app, "_run_child") as run_child:
            self.app._execute(queued_id)
            run_child.assert_not_called()

        timeout_id = "e2f3a4b5c6d7"
        self._write_manual_job(timeout_id)
        with patch.object(
            self.app,
            "_run_child",
            return_value={"returncode": None, "cancelled": False, "timed_out": True},
        ) as run_child:
            self.app._execute(timeout_id)
        timeout_job = self._json(f"/api/runs/{timeout_id}")
        self.assertEqual(timeout_job["status"], "failed")
        self.assertEqual(
            timeout_job["error"],
            "RT 场景准备超过 600 秒，已停止；请减少场景或追踪预算",
        )
        run_child.assert_called_once()

        missing_id = "e3f4a5b6c7d8"
        self._write_manual_job(missing_id)
        with patch.object(
            self.app,
            "_run_child",
            return_value={"returncode": 0, "cancelled": False, "timed_out": False},
        ) as run_child:
            self.app._execute(missing_id)
        missing_job = self._json(f"/api/runs/{missing_id}")
        self.assertEqual(missing_job["status"], "failed")
        self.assertIn("缺少 channel_snapshot.npz", missing_job["error"])
        run_child.assert_called_once()
        self.assertFalse((self.runs_dir / missing_id / "results.csv").exists())

        restart_id = "e4f5a6b7c8d9"
        running = self._write_manual_job(restart_id)
        running["status"] = "running"
        running["stage"] = "tracing"
        running["started_at"] = "2026-10-04T00:00:02+00:00"
        with self.app._lock:
            self.app._write_job(running)
        restored = SimulationWebApp(self.config_dir, self.runs_dir)
        try:
            recovered = restored.get_run(restart_id)
            self.assertEqual(recovered["status"], "failed")
            self.assertEqual(
                recovered["error"],
                "网页服务中断，运行状态已恢复为失败；可重新发起仿真。",
            )
            self.assertTrue((self.runs_dir / restart_id / "scene" / "empty.xml").is_file())
            with self.app._lock:
                self.app._jobs[restart_id].update(
                    {
                        "status": recovered["status"],
                        "stage": recovered["stage"],
                        "finished_at": recovered["finished_at"],
                        "error": recovered["error"],
                    }
                )
        finally:
            restored.shutdown()


    def _write_manual_job(self, job_id: str) -> dict:
        directory = self.runs_dir / job_id
        directory.mkdir(parents=True, exist_ok=False)
        for name, target in (
            ("pusch_4ue.toml", "tx.toml"),
            ("rt_beam_los.toml", "rt.toml"),
            ("bler_rt_beam_web_quick.toml", "simulation.toml"),
        ):
            shutil.copyfile(CONFIGS / name, directory / target)
        rt = RtBeamSettings.from_toml(directory / "rt.toml")
        assets = copy_scene_assets(
            resolve_builtin_scene_assets(rt.rt.scene), directory / "scene"
        )
        job = {
            "id": job_id,
            "status": "queued",
            "channel_backend": "rt",
            "stage": "queued",
            "stage_started_at": None,
            "cancel_requested": False,
            "created_at": "2026-10-04T00:00:00+00:00",
            "started_at": None,
            "finished_at": None,
            "error": None,
            "scene_bundle_sha256": assets.bundle_sha256,
            "scene_source": assets.source,
            "scene_file": assets.scene_file,
            "user_names": [user.name for user in rt.users],
            "config_names": {
                "tx": "pusch_4ue.toml",
                "rt": "rt_beam_los.toml",
                "simulation": "bler_rt_beam_web_quick.toml",
            },
            "total_points": 9,
            "validation_summary": None,
        }
        with self.app._lock:
            self.app._jobs[job_id] = job
            self.app._write_job(job)
        return job

    def test_cancel_before_spawn_and_between_rt_children(self):
        job_id = "a1b2c3d4e5f6"
        job = self._write_manual_job(job_id)
        job["status"] = "running"
        with self.app._lock:
            self.app._write_job(job)
        entered_spawn = threading.Event()
        release_spawn = threading.Event()
        fake = _FakeProcess()

        def blocked_popen(*_args, **_kwargs):
            entered_spawn.set()
            self.assertTrue(release_spawn.wait(10))
            return fake

        outcome = {}
        with patch("nr_pusch.web.subprocess.Popen", side_effect=blocked_popen):
            with patch("nr_pusch.web.threading.Timer") as timer:
                thread = threading.Thread(
                    target=lambda: outcome.update(
                        self.app._run_child(job_id, ["fake-child"])
                    )
                )
                thread.start()
                self.assertTrue(entered_spawn.wait(5))
                cancelled = self.app.cancel_run(job_id)
                self.assertTrue(cancelled["cancel_requested"])
                release_spawn.set()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                self.assertTrue(fake.terminated)
                self.assertTrue(outcome["cancelled"])
                timer.assert_called_once()
                timer.return_value.start.assert_called_once()
        self.app._finish_job(job_id, "cancelled")

        job_id = "b1c2d3e4f5a6"
        self._write_manual_job(job_id)
        directory = self.runs_dir / job_id
        bundle = self.app._jobs[job_id]["scene_bundle_sha256"]
        snapshot_hash = "a" * 64
        (directory / "channel_snapshot.npz").write_bytes(b"snapshot")
        (directory / "channel_snapshot.json").write_text(json.dumps({
            "format_version": 2,
            "array_sha256": snapshot_hash,
            "scene_bundle_sha256": bundle,
            "scene_file": "empty.xml",
            "scene_source": "builtin",
        }))
        (directory / "validation.json").write_text(json.dumps({
            "format_version": 1,
            "policy": "web-frequency-v1",
            "passed": True,
            "strict_fd_td_passed": True,
            "snapshot_hash": snapshot_hash,
            "scene_bundle_sha256": bundle,
            "checks": [],
            "warnings": [],
        }))
        (directory / "snapshot_cache.json").write_text(json.dumps({
            "format_version": 1,
            "cache_key": "b" * 64,
            "hit": False,
        }))
        calls = []

        def prepared_child(_job_id, command, **_kwargs):
            calls.append(command)
            return {"returncode": 0, "cancelled": False, "timed_out": False}

        original_set_stage = self.app._set_stage

        def cancel_at_simulation(_job_id, stage, *, at=None):
            if stage == "simulating":
                self.app.cancel_run(_job_id)
                return False
            return original_set_stage(_job_id, stage, at=at)

        with patch.object(self.app, "_run_child", side_effect=prepared_child):
            with patch.object(self.app, "_set_stage", side_effect=cancel_at_simulation):
                self.app._execute(job_id)
        result = self._json(f"/api/runs/{job_id}")
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("nr_pusch.cli.beam_bler", calls[0])
        self.assertFalse((directory / "results.csv").exists())

    def test_cancel_while_child_is_registered(self):
        job_id = "a2b3c4d5e6f7"
        job = self._write_manual_job(job_id)
        job["status"] = "running"
        with self.app._lock:
            self.app._write_job(job)
        process = _PollingFakeProcess()
        outcome = {}
        with patch("nr_pusch.web.subprocess.Popen", return_value=process):
            with patch("nr_pusch.web.threading.Timer") as timer:
                thread = threading.Thread(
                    target=lambda: outcome.update(
                        self.app._run_child(job_id, ["running-child"])
                    )
                )
                thread.start()
                self.assertTrue(process.polled.wait(5))
                cancelled = self.app.cancel_run(job_id)
                self.assertTrue(cancelled["cancel_requested"])
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                self.assertTrue(process.terminated)
                self.assertTrue(outcome["cancelled"])
                timer.assert_called_once()
        self.app._finish_job(job_id, "cancelled")

    def test_prepare_timeout_terminates_then_kills_stubborn_child(self):
        job_id = "f1a2b3c4d5e6"
        job = self._write_manual_job(job_id)
        job["status"] = "running"
        with self.app._lock:
            self.app._write_job(job)
        process = _StubbornProcess()
        with patch("nr_pusch.web.subprocess.Popen", return_value=process):
            outcome = self.app._run_child(job_id, ["stubborn-child"], timeout_s=0)
        self.assertTrue(outcome["timed_out"])
        self.assertEqual(outcome["returncode"], -9)
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)



class _FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9



class _PollingFakeProcess(_FakeProcess):
    def __init__(self):
        super().__init__()
        self.polled = threading.Event()

    def poll(self):
        self.polled.set()
        return super().poll()
class _StubbornProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if timeout is not None and not self.killed:
            raise subprocess.TimeoutExpired("stubborn-child", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

if __name__ == "__main__":
    unittest.main()
