"""Local web interface for configured BLER runs."""

from __future__ import annotations

import csv
import base64
import binascii
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
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

import numpy as np

from .channel_config import ChannelSettings
from .config import TxSettings
from .simulation_config import BlerSettings
from .dmrs_prior import dmrs_prior_compatibility, resolve_dmrs_tap_power_prior
from .transmitter import NrPuschTx


_PROFILES = {"tx": "pusch_", "channel": "cdl_", "simulation": "bler_"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.toml$")
_SCENARIO_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_RX_CONFIG_NAME = re.compile(r"^rx_[A-Za-z0-9][A-Za-z0-9_.-]*\.toml$")
_DETECTORS = {"lmmse", "lmmse-sic", "k-best", "ep", "mmse-pic", "soft-mmse-pic"}
_DECODE_ID = re.compile(r"^[0-9a-f]{12}$")
_MAX_CAPTURE_BYTES = 12 * 1024 * 1024
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
        self.prior_dir = self.config_dir / "tap_power_prior"
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
        prefixes = {**_PROFILES, "rx": "rx_"}
        if kind not in prefixes or not _NAME.fullmatch(name):
            raise ApiError(404, "配置不存在")
        if not name.startswith(prefixes[kind]):
            raise ApiError(404, "配置类型与文件名不匹配")
        path = self.config_dir / name
        if path.resolve().parent != self.config_dir or not path.is_file():
            raise ApiError(404, "配置不存在")
        return path

    def list_configs(self) -> dict[str, list[str]]:
        configs = {
            kind: sorted(
                path.name for path in self.config_dir.glob(f"{prefix}*.toml")
                if path.is_file() and _NAME.fullmatch(path.name)
            )
            for kind, prefix in _PROFILES.items()
        }
        configs["rx"] = sorted(
            path.name for path in self.config_dir.glob("rx_*.toml")
            if path.is_file() and _RX_CONFIG_NAME.fullmatch(path.name)
        )
        return configs

    def list_scenarios(self) -> list[dict[str, Any]]:
        catalog = self.config_dir / "scenarios.toml"
        if not catalog.is_file():
            return []
        with catalog.open("rb") as stream:
            raw = tomllib.load(stream)
        entries = raw.get("scenarios", [])
        if not isinstance(entries, list):
            raise ValueError("scenarios.toml 必须使用 [[scenarios]] 表")
        scenarios: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("场景条目必须是 TOML 表")
            scenario_id = entry.get("id")
            if (
                not isinstance(scenario_id, str)
                or not _SCENARIO_ID.fullmatch(scenario_id)
                or scenario_id in seen
            ):
                raise ValueError(f"场景 id 无效或重复: {scenario_id!r}")
            label, description = entry.get("label"), entry.get("description")
            profiles = {kind: entry.get(kind) for kind in _PROFILES}
            if not isinstance(label, str) or not label or not isinstance(description, str):
                raise ValueError(f"场景 {scenario_id} 缺少 label/description")
            for kind, name in profiles.items():
                if not isinstance(name, str):
                    raise ValueError(f"场景 {scenario_id} 缺少 {kind} 配置")
                self._profile_path(kind, name)
            seen.add(scenario_id)
            scenarios.append(
                {
                    "id": scenario_id,
                    "label": label,
                    "description": description,
                    "profiles": profiles,
                }
            )
        return scenarios

    def _scenario(self, scenario_id: str) -> dict[str, Any]:
        for scenario in self.list_scenarios():
            if scenario["id"] == scenario_id:
                return scenario
        raise ApiError(400, f"未知实验场景: {scenario_id}")

    def _prepare_run(self, data: dict[str, Any]) -> dict[str, Any]:
        names: dict[str, str] = {}
        texts: dict[str, str] = {}
        errors: list[dict[str, str]] = []
        source_paths: dict[str, Path] = {}
        for kind in _PROFILES:
            name = data.get(f"{kind}_name")
            text = data.get(f"{kind}_text")
            if not isinstance(name, str) or not isinstance(text, str):
                errors.append({"kind": kind, "message": f"缺少 {kind} 配置名称或 TOML 文本"})
                continue
            try:
                source_paths[kind] = self._profile_path(kind, name)
            except ApiError as exc:
                errors.append({"kind": kind, "message": str(exc)})
                continue
            names[kind], texts[kind] = name, text

        scenario_id = data.get("scenario_id") or None
        if scenario_id is not None:
            try:
                scenario = self._scenario(scenario_id)
                if names != scenario["profiles"]:
                    errors.append(
                        {"kind": "scenario", "message": "预设场景的 TX/信道/仿真配置不可拆分修改"}
                    )
                for kind, source_path in source_paths.items():
                    if texts.get(kind) != source_path.read_text(encoding="utf-8"):
                        errors.append(
                            {"kind": kind, "message": "预设配置已被修改；请切换到高级配置模式"}
                        )
            except (ApiError, OSError, ValueError) as exc:
                errors.append({"kind": "scenario", "message": str(exc)})
        if errors:
            return {
                "valid": False,
                "errors": errors,
                "names": names,
                "texts": texts,
                "scenario_id": scenario_id,
            }

        temporary_paths: dict[str, Path] = {}
        settings: dict[str, Any] = {}
        try:
            for kind, text in texts.items():
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".toml",
                    prefix=".web-preflight-",
                    dir=self.config_dir,
                    delete=False,
                ) as stream:
                    stream.write(text)
                    temporary_paths[kind] = Path(stream.name)
            parsers = [
                ("tx", TxSettings),
                ("channel", ChannelSettings),
                ("simulation", BlerSettings),
            ]
            for kind, parser in parsers:
                try:
                    settings[kind] = parser.from_toml(temporary_paths[kind])
                except Exception as exc:
                    errors.append({"kind": kind, "message": str(exc)})
            tx_settings = settings.get("tx")
            channel_settings = settings.get("channel")
            simulation_settings = settings.get("simulation")
            if channel_settings is not None:
                try:
                    channel_settings.validate_transmitter(tx_settings)
                except (ValueError, TypeError) as exc:
                    errors.append({"kind": "channel", "message": str(exc)})
            if tx_settings is not None and channel_settings is not None and simulation_settings is not None:
                stream_count = len(tx_settings.users) * tx_settings.pusch.num_layers
                rx_antennas = (
                    channel_settings.antennas.rx_num_rows
                    * channel_settings.antennas.rx_num_cols
                )
                if "k-best" in simulation_settings.detectors and rx_antennas < stream_count:
                    errors.append(
                        {
                            "kind": "simulation",
                            "message": (
                                f"K-best 要求接收天线数不少于总流数；当前 {rx_antennas} RX / "
                                f"{stream_count} 流"
                            ),
                        }
                    )
                if "dmrs-lmmse" in simulation_settings.channel_estimators_for_sweep:
                    try:
                        max_delay_spread_s = (
                            simulation_settings.max_delay_spread_s
                            if simulation_settings.max_delay_spread_s is not None
                            else channel_settings.channel.max_delay_spread_s
                        )
                        transmitter = NrPuschTx(tx_settings, device="cpu")
                        compatibility = dmrs_prior_compatibility(
                            tx_settings,
                            channel_settings,
                            l_min=simulation_settings.l_min,
                            max_delay_spread_s=max_delay_spread_s,
                            fft_size=transmitter._tx_freq.resource_grid.fft_size,
                            sample_rate_hz=transmitter.sample_rate_hz,
                        )
                        resolve_dmrs_tap_power_prior(
                            self.prior_dir,
                            expected_compatibility=compatibility,
                            device="cpu",
                        )
                    except Exception as exc:
                        errors.append({"kind": "simulation", "message": str(exc)})
        finally:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)

        return {
            "valid": not errors,
            "errors": errors,
            "names": names,
            "texts": texts,
            "scenario_id": scenario_id,
            "settings": settings,
        }


    def validate_run(self, data: dict[str, Any]) -> dict[str, Any]:
        prepared = self._prepare_run(data)
        return {
            "valid": prepared["valid"],
            "errors": prepared["errors"],
            "scenario_id": prepared["scenario_id"],
            "profiles": prepared["names"],
        }

    def compatible_profiles(self, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        if kind not in _PROFILES:
            raise ApiError(400, "未知配置类型")
        results: dict[str, dict[str, Any]] = {}
        for name in self.list_configs()[kind]:
            candidate = dict(data)
            candidate["scenario_id"] = None
            candidate[f"{kind}_name"] = name
            candidate[f"{kind}_text"] = (
                data.get(f"{kind}_text")
                if name == data.get(f"{kind}_name")
                else self.get_config(kind, name)["text"]
            )
            validation = self.validate_run(candidate)
            results[name] = {
                "compatible": validation["valid"],
                "errors": validation["errors"],
            }
        return {"kind": kind, "profiles": results}

    def get_config(self, kind: str, name: str) -> dict[str, str]:
        return {"kind": kind, "name": name, "text": self._profile_path(kind, name).read_text(encoding="utf-8")}

    @staticmethod
    def _validate(kind: str, path: Path) -> None:
        with path.open("rb") as stream:
            tomllib.load(stream)
        {"tx": TxSettings, "rx": TxSettings, "channel": ChannelSettings,
         "simulation": BlerSettings}[kind].from_toml(path)

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
        prepared = self._prepare_run(data)
        if not prepared["valid"]:
            details = "; ".join(
                f'{item["kind"]}: {item["message"]}' for item in prepared["errors"]
            )
            raise ApiError(400, f"配置组合不兼容: {details}")
        names = prepared["names"]
        texts = prepared["texts"]
        simulation_settings: BlerSettings = prepared["settings"]["simulation"]
        job_id = uuid4().hex[:12]
        directory = self._job_dir(job_id)
        directory.mkdir()
        try:
            for kind, text in texts.items():
                path = directory / f"{kind}.toml"
                path.write_text(text, encoding="utf-8")
            job = {
                "id": job_id,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "scenario_id": prepared["scenario_id"],
                "config_names": names,
                "total_points": (
                    len(simulation_settings.channel_estimators_for_sweep)
                    * len(simulation_settings.detectors)
                    * len(simulation_settings.snr_db)
                ),
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
            "--prior-dir", str(self.prior_dir),
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

    def _decode_dir(self, decode_id: str) -> Path:
        if not _DECODE_ID.fullmatch(decode_id):
            raise ApiError(404, "译码记录不存在")
        return self.runs_dir / "rx-decodes" / decode_id

    def rx_defaults(self) -> dict[str, Any]:
        """Describe the repository's ready-to-run local receive example."""
        config_path = self.config_dir / "rx_pusch_4ue.toml"
        capture_path = self.config_dir.parent / "tests" / "fixtures" / "matlab_h5" / "RxTestVector.h5"
        available = config_path.is_file() and capture_path.is_file()
        profile: dict[str, Any] = {}
        if config_path.is_file():
            try:
                profile = tomllib.loads(config_path.read_text(encoding="utf-8")).get("receiver", {})
            except (OSError, tomllib.TOMLDecodeError):
                profile = {}
        if not isinstance(profile, dict):
            profile = {}
        result: dict[str, Any] = {
            "available": available,
            "config_name": config_path.name if config_path.is_file() else None,
            "input_name": capture_path.name if capture_path.is_file() else None,
            "input_format": "matlab-h5",
            "input_domain": profile.get("input_domain", "frequency"),
            "noise_variance": profile.get("noise_variance", 0.0),
            "channel_estimator": profile.get("channel_estimator", "dmrs"),
            "detector": profile.get("detector", "lmmse"),
            "detector_parameter": profile.get("detector_parameter"),
            "detector_damping": profile.get("detector_damping", 0.25),
            "max_delay_spread_s": profile.get("max_delay_spread_s", 3e-6),
        }
        if available:
            result["input_bytes"] = capture_path.stat().st_size
        return result

    def decode_file(self, decode_id: str, name: str) -> Path:
        if name not in {"decoded.npz", "decoded.json"}:
            raise ApiError(404, "译码文件不存在")
        path = self._decode_dir(decode_id) / name
        if not path.is_file():
            raise ApiError(404, "译码文件不存在")
        return path

    def decode_external(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze and decode one uploaded H5 or NPZ capture using the RX CLI."""
        config_name = data.get("config_name")
        config_text = data.get("config_text")
        input_name = data.get("input_name")
        encoded = data.get("input_base64")
        use_default_capture = data.get("use_default_capture", False)
        if not isinstance(use_default_capture, bool):
            raise ApiError(400, "use_default_capture 必须是布尔值")
        default_case = self.rx_defaults()
        default_capture_path = (
            self.config_dir.parent / "tests" / "fixtures" / "matlab_h5" / "RxTestVector.h5"
        )
        if use_default_capture:
            if not default_case["available"]:
                raise ApiError(404, "仓库默认 IQ 夹具或 RX 配置不存在")
            input_name = default_case["input_name"]
            encoded = None
            if not config_name:
                config_name = default_case["config_name"]
            if not config_text:
                config_text = (self.config_dir / str(default_case["config_name"])).read_text(encoding="utf-8")
        if config_name is None:
            config_name = ""
        if not isinstance(config_name, str):
            raise ApiError(400, "config_name 必须是配置名或空字符串")
        config_path: Path | None = None
        if config_name:
            if not _RX_CONFIG_NAME.fullmatch(config_name):
                raise ApiError(400, "RX 配置名必须匹配 rx_*.toml")
            config_path = self.config_dir / config_name
            if config_path.resolve().parent != self.config_dir or not config_path.is_file():
                raise ApiError(404, "接收配置不存在")
        if config_text is None:
            config_text = ""
        if not isinstance(config_text, str):
            raise ApiError(400, "接收配置内容必须是 TOML 文本")
        if not config_text.strip():
            if config_path is None:
                raise ApiError(400, "请粘贴完整 RX TOML 配置，或选择已有配置文件")
            config_text = config_path.read_text(encoding="utf-8")
        profile_receiver: dict[str, Any] = {}
        try:
            loaded = tomllib.loads(config_text).get("receiver", {})
        except tomllib.TOMLDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            profile_receiver = loaded

        def receiver_default(key: str, fallback: Any) -> Any:
            """Resolve like the RX CLI: request wins, then the profile's [receiver], then fallback."""
            value = profile_receiver.get(key, fallback)
            return fallback if value is None else value

        input_format = data.get(
            "input_format", default_case["input_format"] if use_default_capture else None
        )
        input_domain = data.get(
            "input_domain",
            default_case["input_domain"] if use_default_capture
            else receiver_default("input_domain", "time"),
        )
        configured_detector = (
            default_case["detector"] if use_default_capture
            else receiver_default("detector", "lmmse")
        )
        detector = data.get("detector", configured_detector)
        detector_overridden = detector != configured_detector
        channel_estimator = data.get(
            "channel_estimator",
            default_case["channel_estimator"] if use_default_capture
            else receiver_default("channel_estimator", "dmrs"),
        )
        device = data.get("device", "cpu")
        try:
            noise_variance = float(data.get(
                "noise_variance",
                default_case["noise_variance"] if use_default_capture
                else receiver_default("noise_variance", 0.0),
            ))
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "noise_variance 必须是非负数") from exc
        if not math.isfinite(noise_variance) or noise_variance < 0:
            raise ApiError(400, "noise_variance 必须是有限的非负数")
        if not isinstance(input_name, str) or Path(input_name).name != input_name:
            raise ApiError(400, "上传文件名无效")
        suffix = Path(input_name).suffix.lower()
        formats = {".h5": "matlab-h5", ".hdf5": "matlab-h5", ".npz": "npz"}
        if suffix not in formats:
            raise ApiError(400, "仅支持 H5/HDF5 或 NPZ 文件")
        detected_format = formats[suffix]
        if input_format not in {None, "auto", detected_format}:
            raise ApiError(400, "文件扩展名与指定输入格式不匹配")
        if input_domain not in {"time", "frequency"}:
            raise ApiError(400, "input_domain 只支持 time 或 frequency")
        if detected_format == "matlab-h5" and input_domain != "frequency":
            raise ApiError(400, "MATLAB H5 接收夹具是频域网格")
        if detector not in _DETECTORS:
            raise ApiError(400, "不支持的 MIMO 检测算法")
        if device not in {"cpu", "cuda", "cuda:0"}:
            raise ApiError(400, "device 只支持 cpu 或 cuda:0")
        if device == "cuda":
            device = "cuda:0"
        parameter = data.get(
            "detector_parameter",
            None
            if detector_overridden and detector == "soft-mmse-pic"
            else default_case["detector_parameter"] if use_default_capture
            else receiver_default("detector_parameter", None),
        )
        if parameter is not None:
            if isinstance(parameter, bool) or not isinstance(parameter, int) or not 1 <= parameter <= 256:
                raise ApiError(400, "detector_parameter 必须是 1 到 256 的整数")
        try:
            damping = float(data.get(
                "detector_damping",
                0.25
                if detector_overridden and detector == "soft-mmse-pic"
                else default_case["detector_damping"] if use_default_capture
                else receiver_default("detector_damping", 0.25),
            ))
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "detector_damping 必须位于 (0, 1]") from exc
        if not math.isfinite(damping) or not 0 < damping <= 1:
            raise ApiError(400, "detector_damping 必须位于 (0, 1]")
        try:
            max_delay_spread_s = float(data.get(
                "max_delay_spread_s",
                default_case["max_delay_spread_s"] if use_default_capture
                else receiver_default("max_delay_spread_s", 3e-6),
            ))
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "max_delay_spread_s 必须是正数") from exc
        if not math.isfinite(max_delay_spread_s) or max_delay_spread_s <= 0:
            raise ApiError(400, "max_delay_spread_s 必须是正数")
        rx_channel_config: Path | None = None
        if channel_estimator == "dmrs-lmmse":
            channel_config_name = data.get("channel_config_name")
            if not isinstance(channel_config_name, str) or not channel_config_name:
                raise ApiError(400, "dmrs-lmmse 需要选择与采集匹配的信道配置")
            rx_channel_config = self._profile_path("channel", channel_config_name)
            if "dmrs_tap_power_prior_path" in profile_receiver:
                raise ApiError(
                    400,
                    "dmrs_tap_power_prior_path 已移除；先验改为按信道配置在共享先验目录中自动查找",
                )
        if use_default_capture:
            payload = default_capture_path.read_bytes()
        else:
            if not isinstance(encoded, str):
                raise ApiError(400, "缺少上传文件数据")
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ApiError(400, "上传文件编码无效") from exc
        if not payload or len(payload) > _MAX_CAPTURE_BYTES:
            raise ApiError(413, "上传文件为空或超过 12 MiB")
        scrambling_name = data.get("scrambling_name")
        scrambling_encoded = data.get("scrambling_base64")
        if scrambling_encoded is not None and not isinstance(scrambling_encoded, str):
            raise ApiError(400, "加扰序列文件编码无效")
        scrambling_payload: bytes | None = None
        if scrambling_encoded:
            try:
                scrambling_payload = base64.b64decode(scrambling_encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ApiError(400, "加扰序列文件编码无效") from exc
            if not scrambling_payload or len(scrambling_payload) > _MAX_CAPTURE_BYTES:
                raise ApiError(413, "加扰序列文件为空或超过 12 MiB")
            if not isinstance(scrambling_name, str) or Path(scrambling_name).name != scrambling_name:
                raise ApiError(400, "加扰序列文件名无效")
            if Path(scrambling_name).suffix.lower() not in {".h5", ".hdf5"}:
                raise ApiError(400, "加扰序列仅支持 H5/HDF5 文件")
        else:
            scrambling_name = None

        decode_id = uuid4().hex[:12]
        directory = self._decode_dir(decode_id)
        directory.mkdir(parents=True, exist_ok=False)
        try:
            config_copy = directory / "rx.toml"
            capture = directory / f"capture{suffix}"
            output = directory / "decoded.npz"
            config_copy.write_text(config_text, encoding="utf-8")
            try:
                self._validate("rx", config_copy)
            except (OSError, ValueError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
                raise ApiError(400, f"接收配置无效: {exc}") from exc
            if rx_channel_config is not None:
                try:
                    rx_tx_settings = TxSettings.from_toml(config_copy)
                    rx_channel_settings = ChannelSettings.from_toml(rx_channel_config)
                    rx_channel_settings.validate_transmitter(rx_tx_settings)
                    rx_transmitter = NrPuschTx(rx_tx_settings, device="cpu")
                    rx_compatibility = dmrs_prior_compatibility(
                        rx_tx_settings,
                        rx_channel_settings,
                        l_min=int(receiver_default("l_min", -6)),
                        max_delay_spread_s=max_delay_spread_s,
                        fft_size=rx_transmitter._tx_freq.resource_grid.fft_size,
                        sample_rate_hz=rx_transmitter.sample_rate_hz,
                    )
                    resolve_dmrs_tap_power_prior(
                        self.prior_dir,
                        expected_compatibility=rx_compatibility,
                        device="cpu",
                    )
                except Exception as exc:
                    raise ApiError(422, f"无法加载匹配的 DMRS tap-power prior: {exc}") from exc
            capture.write_bytes(payload)
            if rx_channel_config is not None:
                channel_copy = directory / "channel.toml"
                channel_copy.write_bytes(rx_channel_config.read_bytes())
                self._validate("channel", channel_copy)
            scrambling_copy: Path | None = None
            if scrambling_payload is not None:
                scrambling_copy = directory / f"scrambling{Path(str(scrambling_name)).suffix.lower()}"
                scrambling_copy.write_bytes(scrambling_payload)
            command = [
                self.python, "-u", "-m", "nr_pusch.cli.rx",
                "--rx-config", str(config_copy), "--input", str(capture),
                "--input-format", detected_format,
                "--noise-variance", str(noise_variance),
                "--channel-estimator", channel_estimator, "--input-domain",
                input_domain,
                "--detector", detector, "--detector-damping", str(damping),
                "--max-delay-spread-s", str(max_delay_spread_s),
                "--output", str(output), "--device", device,
                "--cb-crc",
            ]
            if rx_channel_config is not None:
                command.extend(
                    (
                        "--channel-config", str(directory / "channel.toml"),
                        "--prior-dir", str(self.prior_dir),
                    )
                )
            if parameter is not None:
                command.extend(("--detector-parameter", str(parameter)))
            if scrambling_copy is not None:
                command.extend(("--scrambling", str(scrambling_copy)))
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if completed.returncode != 0 or not output.is_file():
                detail = (completed.stdout + "\n" + completed.stderr).strip()[-3000:]
                raise ApiError(422, f"接收处理失败: {detail or '解码器没有生成结果'}")
            manifest_path = output.with_suffix(".json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with np.load(output, allow_pickle=False) as result:
                crc = np.asarray(result["crc_status"], dtype=np.bool_)
                bits_shape = list(result["bits"].shape)
                real = np.asarray(result["constellation_real"], dtype=np.float32)
                imag = np.asarray(result["constellation_imag"], dtype=np.float32)
            crc_flat = crc.reshape(-1).tolist()
            configured_users = manifest.get("settings", {}).get("users", [])
            user_names = [
                str(user.get("name", f"UE{index}"))
                for index, user in enumerate(configured_users)
                if isinstance(user, dict)
            ] or [f"UE{index}" for index in range(crc.shape[-1])]
            constellation_users: list[dict[str, Any]] = []
            for user_index in range(real.shape[1]):
                for layer_index in range(real.shape[2]):
                    user_real = real[:, user_index, layer_index, :].reshape(-1)
                    user_imag = imag[:, user_index, layer_index, :].reshape(-1)
                    sample_indices = np.linspace(
                        0, user_real.size - 1, min(1500, user_real.size), dtype=np.int64
                    )
                    constellation_users.append({
                        "name": (
                            user_names[user_index % len(user_names)]
                            if real.shape[2] == 1 else
                            f"{user_names[user_index % len(user_names)]} · layer {layer_index}"
                        ),
                        "user": user_index,
                        "layer": layer_index,
                        "real": user_real[sample_indices].tolist(),
                        "imag": user_imag[sample_indices].tolist(),
                    })
            crc_labels = [
                user_names[index % len(user_names)]
                if crc.ndim < 2 or crc.shape[0] == 1
                else f"Frame {index // len(user_names) + 1} · {user_names[index % len(user_names)]}"
                for index in range(len(crc_flat))
            ]
            receiver_metadata = manifest.get("receiver", {})
            cb_crc_users: list[dict[str, Any]] = []
            cb_crc_rows = receiver_metadata.get("cb_crc_status")
            if isinstance(cb_crc_rows, list):
                for user_index, row in enumerate(cb_crc_rows):
                    if not isinstance(row, list):
                        continue
                    status = [bool(value) for value in row]
                    cb_crc_users.append({
                        "name": user_names[user_index % len(user_names)],
                        "status": status,
                        "passed": sum(status),
                        "total": len(status),
                    })
            return {
                "decode_id": decode_id,
                "input": {
                    "name": input_name,
                    "format": detected_format,
                    "source": "repository-default" if use_default_capture else "upload",
                    "bytes": len(payload),
                    "scrambling": scrambling_name,
                    **manifest.get("input_analysis", {}),
                },
                "crc_status": crc_flat,
                "crc_labels": crc_labels,
                "crc_pass_count": int(crc.sum()),
                "block_count": int(crc.size),
                "bits_shape": bits_shape,
                "constellation": {"users": constellation_users},
                "detector": detector,
                "used_default_capture": use_default_capture,
                "cb_crc": {"available": bool(cb_crc_users), "users": cb_crc_users},
                "receiver": receiver_metadata,
                "reference_comparison": manifest.get("reference_comparison"),
                "output_npz_url": f"/api/rx/decode/{decode_id}/decoded.npz",
                "output_json_url": f"/api/rx/decode/{decode_id}/decoded.json",
                "logs": completed.stdout.splitlines()[-12:],
            }
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise ApiError(504, "译码超过 5 分钟，已终止") from exc
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

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

        def _body(self, max_bytes: int = 512_000) -> dict[str, Any]:
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ApiError(400, "Content-Length 无效") from exc
            if not 0 < size <= max_bytes:
                raise ApiError(413, f"请求体为空或超过 {max_bytes // 1024} KiB")
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
                elif method == "GET" and path == "/api/scenarios":
                    self._json(200, {"scenarios": app.list_scenarios()})
                elif method == "POST" and path == "/api/validate-run":
                    self._json(200, app.validate_run(self._body()))
                elif method == "POST" and path == "/api/profile-compatibility":
                    body = self._body()
                    self._json(200, app.compatible_profiles(body.get("kind"), body))
                elif method == "GET" and path == "/api/rx/configs":
                    self._json(200, {"configs": app.list_configs()["rx"]})
                elif method == "GET" and path == "/api/rx/defaults":
                    self._json(200, app.rx_defaults())
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
                elif path == "/api/rx/decode" and method == "POST":
                    self._json(200, app.decode_external(self._body(max_bytes=18 * 1024 * 1024)))
                elif len(parts) == 5 and parts[:3] == ["api", "rx", "decode"] and method == "GET":
                    file = app.decode_file(parts[3], parts[4])
                    mime = "application/json; charset=utf-8" if file.suffix == ".json" else "application/octet-stream"
                    self._send(200, file.read_bytes(), mime)
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
