"""Local web interface for configured BLER runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tomllib
import traceback
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from .channel_config import ChannelSettings
from .config import TxSettings
from .simulation_config import BlerSettings


_PROFILES = {"tx": "pusch_", "channel": "cdl_", "simulation": "bler_"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.toml$")
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class SimulationWebApp:
    """Own profile access, a single simulation worker, and persistent run metadata."""

    def __init__(self, config_dir: Path, runs_dir: Path, python: str = sys.executable):
        self.config_dir = config_dir.resolve()
        self.runs_dir = runs_dir.resolve()
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.python = python
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._pending: queue.Queue[str] = queue.Queue()
        self._load_jobs()
        threading.Thread(target=self._work, name="bler-web-worker", daemon=True).start()

    def _load_jobs(self) -> None:
        for path in self.runs_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if not _JOB_ID.fullmatch(job["id"]):
                    continue
                if job["status"] in {"queued", "running"}:
                    job["status"] = "failed"
                    job["finished_at"] = _now()
                    job["error"] = "网页服务中断，运行状态已恢复为失败；可重新发起仿真。"
                    self._write_job(job)
                self._jobs[job["id"]] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def _profile_path(self, kind: str, name: str) -> Path:
        if kind not in _PROFILES or not _NAME.fullmatch(name):
            raise ApiError(404, "配置不存在")
        if not name.startswith(_PROFILES[kind]):
            raise ApiError(404, "配置类型与文件名不匹配")
        path = self.config_dir / name
        if path.resolve().parent != self.config_dir or not path.is_file():
            raise ApiError(404, "配置不存在")
        return path

    def list_configs(self) -> dict[str, list[str]]:
        return {
            kind: sorted(
                path.name for path in self.config_dir.glob(f"{prefix}*.toml")
                if path.is_file() and _NAME.fullmatch(path.name)
            )
            for kind, prefix in _PROFILES.items()
        }

    def get_config(self, kind: str, name: str) -> dict[str, str]:
        return {"kind": kind, "name": name, "text": self._profile_path(kind, name).read_text(encoding="utf-8")}

    @staticmethod
    def _validate(kind: str, path: Path) -> None:
        with path.open("rb") as stream:
            tomllib.load(stream)
        {"tx": TxSettings, "channel": ChannelSettings, "simulation": BlerSettings}[kind].from_toml(path)

    def save_config(self, kind: str, name: str, text: str) -> dict[str, str]:
        destination = self._profile_path(kind, name)
        if not isinstance(text, str):
            raise ApiError(400, "text 必须为 TOML 文本")
        candidate: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".toml", prefix=".web-",
                dir=self.config_dir, delete=False,
            ) as stream:
                candidate = Path(stream.name)
                stream.write(text)
            self._validate(kind, candidate)
            os.replace(candidate, destination)
        finally:
            if candidate is not None:
                candidate.unlink(missing_ok=True)
        return {"kind": kind, "name": name, "text": text}

    def _job_dir(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise ApiError(404, "运行不存在")
        return self.runs_dir / job_id

    def _write_job(self, job: dict[str, Any]) -> None:
        path = self._job_dir(job["id"]) / "job.json"
        staging = path.with_suffix(".json.tmp")
        staging.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, path)

    def create_run(self, data: dict[str, Any]) -> dict[str, Any]:
        names: dict[str, str] = {}
        texts: dict[str, str] = {}
        for kind in _PROFILES:
            name = data.get(f"{kind}_name")
            text = data.get(f"{kind}_text")
            if not isinstance(name, str) or not isinstance(text, str):
                raise ApiError(400, f"缺少 {kind} 配置名称或 TOML 文本")
            self._profile_path(kind, name)
            names[kind], texts[kind] = name, text

        job_id = uuid4().hex[:12]
        directory = self._job_dir(job_id)
        directory.mkdir()
        try:
            simulation_settings: BlerSettings | None = None
            for kind, text in texts.items():
                path = directory / f"{kind}.toml"
                path.write_text(text, encoding="utf-8")
                self._validate(kind, path)
                if kind == "simulation":
                    simulation_settings = BlerSettings.from_toml(path)
            assert simulation_settings is not None
            job = {
                "id": job_id,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "config_names": names,
                "total_points": len(simulation_settings.detectors) * len(simulation_settings.snr_db),
            }
            with self._lock:
                self._jobs[job_id] = job
                self._write_job(job)
                self._pending.put(job_id)
            return self.get_run(job_id)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def _work(self) -> None:
        while True:
            job_id = self._pending.get()
            try:
                self._execute(job_id)
            except Exception as exc:
                traceback.print_exc()
                with self._lock:
                    job = self._jobs[job_id]
                    job["status"] = "failed"
                    job["error"] = f"任务调度失败: {exc}"
                    job["finished_at"] = _now()
                    self._write_job(job)
            finally:
                self._pending.task_done()

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] != "queued":
                return
            job["status"] = "running"
            job["started_at"] = _now()
            self._write_job(job)
        directory = self._job_dir(job_id)
        command = [
            self.python, "-u", "-m", "nr_pusch.cli.bler",
            "--tx-config", str(directory / "tx.toml"),
            "--channel-config", str(directory / "channel.toml"),
            "--simulation-config", str(directory / "simulation.toml"),
            "--output", str(directory / "results.csv"),
            "--progress-jsonl", str(directory / "progress.jsonl"),
        ]
        try:
            with (directory / "run.log").open("wb") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                with self._lock:
                    self._processes[job_id] = process
                    if job.get("cancel_requested"):
                        process.terminate()
                returncode = process.wait()
            with self._lock:
                self._processes.pop(job_id, None)
                if job.get("cancel_requested"):
                    job["status"] = "cancelled"
                elif returncode == 0 and (directory / "results.csv").is_file():
                    job["status"] = "completed"
                else:
                    job["status"] = "failed"
                    job["error"] = f"仿真进程退出码 {returncode}。请查看运行日志。"
                job["finished_at"] = _now()
                self._write_job(job)
        except Exception as exc:
            with self._lock:
                self._processes.pop(job_id, None)
                job["status"] = "failed"
                job["error"] = f"无法启动仿真进程: {exc}"
                job["finished_at"] = _now()
                self._write_job(job)

    def cancel_run(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, "运行不存在")
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["finished_at"] = _now()
            elif job["status"] == "running":
                job["cancel_requested"] = True
                process = self._processes.get(job_id)
                if process is not None and process.poll() is None:
                    process.terminate()
                    threading.Timer(5, self._kill_if_running, args=(process,)).start()
            self._write_job(job)
        return self.get_run(job_id)

    @staticmethod
    def _kill_if_running(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.kill()

    def _points(self, job_id: str) -> list[dict[str, Any]]:
        directory = self._job_dir(job_id)
        progress = directory / "progress.jsonl"
        if progress.is_file():
            points: list[dict[str, Any]] = []
            for line in progress.read_text(encoding="utf-8").splitlines():
                try:
                    points.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return points
        results = directory / "results.csv"
        if results.is_file():
            with results.open(newline="", encoding="utf-8") as stream:
                return list(csv.DictReader(stream))
        return []

    def get_run(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, "运行不存在")
            result = dict(job)
        directory = self._job_dir(job_id)
        log = directory / "run.log"
        result["points"] = self._points(job_id)
        result["logs"] = log.read_text(encoding="utf-8", errors="replace").splitlines()[-100:] if log.is_file() else []
        result["output_csv"] = f"/api/runs/{job_id}/results.csv" if (directory / "results.csv").is_file() else None
        result["output_manifest"] = f"/api/runs/{job_id}/results.json" if (directory / "results.json").is_file() else None
        return result

    def list_runs(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            ids = sorted(self._jobs, key=lambda key: self._jobs[key]["created_at"], reverse=True)
        return {"runs": [self.get_run(job_id) for job_id in ids]}

    def result_file(self, job_id: str, name: str) -> Path:
        self.get_run(job_id)
        if name not in {"results.csv", "results.json"}:
            raise ApiError(404, "结果文件不存在")
        path = self._job_dir(job_id) / name
        if not path.is_file():
            raise ApiError(404, "结果文件不存在")
        return path

    def shutdown(self) -> None:
        """Stop child simulations before the web process exits."""
        with self._lock:
            processes = list(self._processes.values())
            for job in self._jobs.values():
                if job["status"] == "queued":
                    job["status"] = "cancelled"
                    job["finished_at"] = _now()
                    self._write_job(job)
                elif job["status"] == "running":
                    job["cancel_requested"] = True
                    self._write_job(job)
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def make_handler(app: SimulationWebApp) -> type[BaseHTTPRequestHandler]:
    static_dir = Path(__file__).with_name("web_static")

    class Handler(BaseHTTPRequestHandler):
        server_version = "NrPuschWeb/0.1"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: Any) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self) -> dict[str, Any]:
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ApiError(400, "Content-Length 无效") from exc
            if not 0 < size <= 512_000:
                raise ApiError(413, "请求体为空或超过 500 KiB")
            if self.headers.get_content_type() != "application/json":
                raise ApiError(415, "请求必须使用 application/json")
            try:
                value = json.loads(self.rfile.read(size))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ApiError(400, "JSON 请求体无效") from exc
            if not isinstance(value, dict):
                raise ApiError(400, "请求体必须为 JSON 对象")
            return value

        def _handle(self, method: str) -> None:
            try:
                path = urlsplit(self.path).path
                parts = path.strip("/").split("/")
                if method == "GET" and path in _ASSETS:
                    name, mime = _ASSETS[path]
                    self._send(200, (static_dir / name).read_bytes(), mime)
                elif method == "GET" and path == "/api/configs":
                    self._json(200, app.list_configs())
                elif len(parts) == 4 and parts[:2] == ["api", "configs"] and method in {"GET", "PUT"}:
                    kind, name = parts[2:]
                    if method == "GET":
                        self._json(200, app.get_config(kind, name))
                    else:
                        self._json(200, app.save_config(kind, name, self._body().get("text")))
                elif path == "/api/runs" and method == "GET":
                    self._json(200, app.list_runs())
                elif path == "/api/runs" and method == "POST":
                    self._json(HTTPStatus.CREATED, app.create_run(self._body()))
                elif len(parts) == 3 and parts[:2] == ["api", "runs"] and method == "GET":
                    self._json(200, app.get_run(parts[2]))
                elif len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel" and method == "POST":
                    self._json(200, app.cancel_run(parts[2]))
                elif len(parts) == 4 and parts[:2] == ["api", "runs"] and method == "GET":
                    file = app.result_file(parts[2], parts[3])
                    mime = "text/csv; charset=utf-8" if file.suffix == ".csv" else "application/json; charset=utf-8"
                    self._send(200, file.read_bytes(), mime)
                else:
                    raise ApiError(404, "接口不存在")
            except ApiError as exc:
                self._json(exc.status, {"error": str(exc)})
            except (OSError, ValueError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
                self._json(400, {"error": f"配置或请求无效: {exc}"})
            except Exception:
                traceback.print_exc()
                self._json(500, {"error": "服务内部错误；请查看终端日志"})

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def do_PUT(self) -> None:
            self._handle("PUT")

    return Handler


def serve(host: str, port: int, config_dir: Path, runs_dir: Path) -> None:
    app = SimulationWebApp(config_dir, runs_dir)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    print(f"NR PUSCH 仿真界面: http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        server.server_close()
