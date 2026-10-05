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
import time
import tomllib
import traceback
import zipfile
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from .channel_config import ChannelSettings
from .config import TxSettings
from .dmrs_prior import dmrs_prior_compatibility, resolve_dmrs_tap_power_prior
from .rt_config import RtBeamSettings, validate_rt_web_limits
from .rt_scene_assets import (
    RtSceneAssets,
    ValidatedRtSceneBundle,
    build_parameterized_scene_bundle,
    copy_scene_assets,
    make_scene_template_zip,
    parse_scene_bundle_zip,
    persist_scene_bundle,
    resolve_builtin_scene_assets,
    resolve_scene_assets,
    builtin_scene_catalog,
)
from .rt_scene_preview import render_rt_scene_preview
from .simulation_config import BlerSettings
from .transmitter import NrPuschTx


_PROFILES = {"tx": "pusch_", "channel": "cdl_", "rt": "rt_beam_", "simulation": "bler_"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.toml$")
_SCENARIO_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_SCENE_ID = re.compile(r"^[0-9a-f]{12}$")
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
        self.outputs_dir = Path(__file__).resolve().parents[2] / "outputs"
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.scene_store_dir = self.runs_dir / "_rt_scenes"
        if self.scene_store_dir.is_symlink():
            raise RuntimeError("RT 场景存储目录不得是符号链接")
        self.scene_store_dir.mkdir(mode=0o700, exist_ok=True)
        if self.scene_store_dir.resolve().parent != self.runs_dir:
            raise RuntimeError("RT 场景存储目录必须直接位于 runs 目录下")
        os.chmod(self.scene_store_dir, 0o700)
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
            backend = entry.get("channel_backend", "cdl")
            if backend not in {"cdl", "rt"}:
                raise ValueError(f"场景 {scenario_id} channel_backend 无效")
            profile_kinds = ("tx", "channel", "simulation") if backend == "cdl" else ("tx", "rt", "simulation")
            forbidden_kinds = {"rt"} if backend == "cdl" else {"channel"}
            if any(entry.get(kind) is not None for kind in forbidden_kinds):
                raise ValueError(f"场景 {scenario_id} 混用了 {backend} profile")
            profiles = {kind: entry.get(kind) for kind in profile_kinds}
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
                    "channel_backend": backend,
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
        backend = data.get("channel_backend", "cdl")
        errors: list[dict[str, str]] = []
        if backend not in {"cdl", "rt"}:
            errors.append({"kind": "channel_backend", "message": "channel_backend 必须是 cdl 或 rt"})
            backend = "cdl"
        kinds = ("tx", "channel", "simulation") if backend == "cdl" else ("tx", "rt", "simulation")
        forbidden = ("rt_name", "rt_text", "scene_id") if backend == "cdl" else ("channel_name", "channel_text")
        for field in forbidden:
            if data.get(field) is not None:
                errors.append({"kind": field.split("_")[0], "message": f"{backend} 请求不得携带 {field}"})

        names: dict[str, str] = {}
        texts: dict[str, str] = {}
        source_paths: dict[str, Path] = {}
        for kind in kinds:
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
            if not isinstance(scenario_id, str):
                errors.append({"kind": "scenario", "message": "scenario_id 必须是字符串"})
            else:
                try:
                    scenario = self._scenario(scenario_id)
                    if scenario["channel_backend"] != backend or names != scenario["profiles"]:
                        errors.append(
                            {"kind": "scenario", "message": "预设场景的 backend 与 TX/信道/仿真配置不可拆分修改"}
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
                "valid": False, "errors": errors, "names": names, "texts": texts,
                "scenario_id": scenario_id, "channel_backend": backend,
            }

        temporary_paths: dict[str, Path] = {}
        settings: dict[str, Any] = {}
        scene_assets: RtSceneAssets | None = None
        scene_bundle: ValidatedRtSceneBundle | None = None
        try:
            for kind, text in texts.items():
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".toml", prefix=".web-preflight-",
                    dir=self.config_dir, delete=False,
                ) as stream:
                    stream.write(text)
                    temporary_paths[kind] = Path(stream.name)
            parsers = {
                "tx": TxSettings,
                "channel": ChannelSettings,
                "rt": RtBeamSettings,
                "simulation": BlerSettings,
            }
            for kind in kinds:
                try:
                    settings[kind] = parsers[kind].from_toml(temporary_paths[kind])
                except Exception as exc:
                    errors.append({"kind": kind, "message": str(exc)})

            tx_settings = settings.get("tx")
            simulation_settings = settings.get("simulation")
            if backend == "cdl":
                channel_settings = settings.get("channel")
                if channel_settings is not None:
                    try:
                        channel_settings.validate_transmitter(tx_settings)
                    except (ValueError, TypeError) as exc:
                        errors.append({"kind": "channel", "message": str(exc)})
                if tx_settings is not None and channel_settings is not None and simulation_settings is not None:
                    stream_count = len(tx_settings.users) * tx_settings.pusch.num_layers
                    rx_antennas = channel_settings.antennas.rx_num_rows * channel_settings.antennas.rx_num_cols
                    if "k-best" in simulation_settings.detectors and rx_antennas < stream_count:
                        errors.append({
                            "kind": "simulation",
                            "message": f"K-best 要求接收天线数不少于总流数；当前 {rx_antennas} RX / {stream_count} 流",
                        })
                    if "dmrs-lmmse" in simulation_settings.channel_estimators_for_sweep:
                        try:
                            max_delay_spread_s = (
                                simulation_settings.max_delay_spread_s
                                if simulation_settings.max_delay_spread_s is not None
                                else channel_settings.channel.max_delay_spread_s
                            )
                            transmitter = NrPuschTx(tx_settings, device="cpu")
                            compatibility = dmrs_prior_compatibility(
                                tx_settings, channel_settings, l_min=simulation_settings.l_min,
                                max_delay_spread_s=max_delay_spread_s,
                                fft_size=transmitter._tx_freq.resource_grid.fft_size,
                                sample_rate_hz=transmitter.sample_rate_hz,
                            )
                            resolve_dmrs_tap_power_prior(
                                self.prior_dir, expected_compatibility=compatibility, device="cpu"
                            )
                        except Exception as exc:
                            errors.append({"kind": "simulation", "message": str(exc)})
            else:
                rt_settings = settings.get("rt")
                if tx_settings is not None and rt_settings is not None:
                    try:
                        rt_settings.validate_transmitter(tx_settings)
                        validate_rt_web_limits(rt_settings, simulation_settings)
                    except (ValueError, TypeError) as exc:
                        errors.append({"kind": "rt", "message": str(exc)})
                scene_id = data.get("scene_id") or None
                if scene_id is not None and not isinstance(scene_id, str):
                    errors.append({"kind": "scene_id", "field": "scene_id", "message": "scene_id 必须是字符串"})
                elif rt_settings is not None:
                    try:
                        scene_assets, scene_bundle = self._resolve_rt_scene(rt_settings, scene_id)
                    except (OSError, ValueError, TypeError) as exc:
                        field = "scene_id" if rt_settings.rt.scene == "custom" else "geometry"
                        errors.append({"kind": "scene", "field": field, "message": str(exc)})
        finally:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)

        return {
            "valid": not errors, "errors": errors, "names": names, "texts": texts,
            "scenario_id": scenario_id, "channel_backend": backend, "settings": settings,
            "scene_assets": scene_assets, "scene_bundle": scene_bundle,
        }


    def validate_run(self, data: dict[str, Any]) -> dict[str, Any]:
        prepared = self._prepare_run(data)
        settings = prepared.get("settings", {})
        simulation = settings.get("simulation")
        total_points = (
            len(simulation.channel_estimators_for_sweep)
            * len(simulation.detectors)
            * len(simulation.snr_db)
            if simulation is not None and prepared["valid"]
            else None
        )
        return {
            "valid": prepared["valid"],
            "errors": prepared["errors"],
            "warnings": [],
            "scenario_id": prepared["scenario_id"],
            "channel_backend": prepared["channel_backend"],
            "profiles": prepared["names"],
            "total_points": total_points,
        }

    def compatible_profiles(self, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        if kind not in _PROFILES:
            raise ApiError(400, "未知配置类型")
        results: dict[str, dict[str, Any]] = {}
        for name in self.list_configs()[kind]:
            candidate = dict(data)
            candidate["scenario_id"] = None
            if kind == "rt":
                candidate["channel_backend"] = "rt"
                candidate.pop("channel_name", None)
                candidate.pop("channel_text", None)
            elif kind == "channel":
                candidate["channel_backend"] = "cdl"
                candidate.pop("rt_name", None)
                candidate.pop("rt_text", None)
                candidate.pop("scene_id", None)
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
    def rt_config(self, data: dict[str, Any]) -> dict[str, Any]:
        has_object = "rt_settings" in data
        has_text = "rt_text" in data
        if has_object == has_text:
            raise ApiError(400, "rt_settings 与 rt_text 必须且只能提供一个")
        try:
            if has_object:
                raw = data["rt_settings"]
                if not isinstance(raw, dict):
                    raise ValueError("rt_settings 必须是对象")
                settings = RtBeamSettings.from_dict(raw)
            else:
                text = data["rt_text"]
                if not isinstance(text, str):
                    raise ValueError("rt_text 必须是 TOML 文本")
                settings = RtBeamSettings.from_dict(tomllib.loads(text))
            validate_rt_web_limits(settings)
        except (ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
            raise ApiError(400, f"RT 配置无效: {exc}") from exc
        receiver = settings.receiver
        return {
            "rt_settings": settings.to_dict(),
            "rt_text": settings.to_toml(),
            "summary": {
                "scene": settings.rt.scene,
                "carrier_frequency_hz": settings.rt.carrier_frequency_hz,
                "receiver_array": {
                    "rows": receiver.num_rows,
                    "cols": receiver.num_cols,
                    "elements": receiver.num_rows * receiver.num_cols,
                },
                "user_count": len(settings.users),
                "post_combiner_ratio": settings.noise.post_combiner_ratio,
                "max_depth": settings.rt.max_depth,
                "samples_per_src": settings.rt.samples_per_src,
                "max_num_paths_per_src": settings.rt.max_num_paths_per_src,
                "gate_policy": "web-frequency-v1",
            },
        }

    def rt_options(self) -> dict[str, Any]:
        devices = ["cpu"]
        warning = None
        try:
            import torch

            if torch.cuda.is_available():
                devices.append("cuda:0")
            else:
                warning = "PyTorch CUDA 不可用；较长统计配置将禁用，可在专家 TOML 中显式选择 CPU。"
        except Exception as exc:
            warning = f"无法查询 PyTorch CUDA 能力：{exc}"
        return {
            "scene_presets": builtin_scene_catalog(),
            "devices": devices,
            "web_limits": {
                "carrier_frequency_hz": [1e8, 1e11],
                "max_depth": [0, 3],
                "samples_per_src": [10_000, 200_000],
                "max_num_paths_per_src": [1, 10_000],
                "array_elements": [4, 64],
                "batch_size": [1, 20],
                "max_frames_per_snr": [1, 2_000],
                "stop_at_zero_bler": [False, True],
                "snr_points": [1, 32],
                "snr_db": [-30.0, 100.0],
            },
            "model_constraints": {
                "users": ["ue0", "ue1", "ue2", "ue3"],
                "layers_per_user": 1,
                "tx_ports_per_user": 1,
                "synthetic_array": True,
                "array_phase_approximation": "载频阵列相位近似，不建模 beam squint",
                "propagation": "静态 LoS＋specular；无折射/散射/衍射/多普勒",
            },
            "gate_policy": {
                "id": "web-frequency-v1",
                "max_fd_td_relative_rms": 0.001,
                "max_truncation_relative_error": 0.01,
                "description": "固定几何/静态快照的条件频域 BLER；strict 1e-5 另行标记。",
            },
            "long_statistics_enabled": "cuda:0" in devices,
            "warning": warning,
        }

    def upload_scene(self, data: dict[str, Any]) -> dict[str, Any]:
        filename = data.get("filename")
        encoded = data.get("content_base64")
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.lower().endswith(".zip"):
            raise ApiError(400, "filename 必须是单个 .zip 文件名")
        if not isinstance(encoded, str):
            raise ApiError(400, "content_base64 必须是 Base64 字符串")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ApiError(400, "场景 ZIP Base64 编码无效") from exc
        if not payload:
            raise ApiError(400, "场景 ZIP 不能为空")
        try:
            bundle = parse_scene_bundle_zip(payload)
        except (ValueError, TypeError) as exc:
            raise ApiError(400, f"场景包无效: {exc}") from exc
        for _ in range(3):
            scene_id = uuid4().hex[:12]
            destination = self.scene_store_dir / scene_id
            try:
                assets = persist_scene_bundle(bundle, destination, source="imported")
                break
            except FileExistsError:
                continue
        else:
            raise ApiError(503, "无法分配场景 ID")
        return {
            "scene_id": scene_id,
            "bundle_sha256": assets.bundle_sha256,
            "files": assets.file_sha256,
            "mesh_summary": bundle.mesh_summary,
        }

    def _resolve_rt_scene(
        self, settings: RtBeamSettings, scene_id: str | None
    ) -> tuple[RtSceneAssets | None, ValidatedRtSceneBundle | None]:
        if settings.rt.scene == "custom":
            if scene_id is None or not _SCENE_ID.fullmatch(scene_id):
                raise ValueError("custom 场景必须选择一个有效 scene_id")
            assets = resolve_scene_assets(
                self.scene_store_dir / scene_id,
                settings.rt.scene_file or "scene.xml",
                source="imported",
            )
            return assets, None
        if scene_id is not None:
            raise ValueError("scene_id 仅适用于 custom 场景")
        if settings.geometry is not None:
            bundle = build_parameterized_scene_bundle(settings.rt.scene, settings.geometry)
            return None, bundle
        return resolve_builtin_scene_assets(settings.rt.scene), None

    def rt_scene_preview(self, data: dict[str, Any]) -> bytes:
        unknown = set(data).difference({"rt_settings", "scene_id", "view"})
        if unknown:
            raise ApiError(400, f"场景预览包含未知字段: {sorted(unknown)}")
        raw_settings = data.get("rt_settings")
        if not isinstance(raw_settings, dict):
            raise ApiError(400, "rt_settings 必须是 RT 配置对象")
        scene_id = data.get("scene_id")
        if scene_id is not None and not isinstance(scene_id, str):
            raise ApiError(400, "scene_id 必须是字符串")
        view = data.get("view", "oblique")
        if not isinstance(view, str) or view not in {"oblique", "top"}:
            raise ApiError(400, "view 必须是 oblique 或 top")

        settings = RtBeamSettings.from_dict(raw_settings)
        try:
            assets, bundle = self._resolve_rt_scene(settings, scene_id)
            if bundle is not None:
                with tempfile.TemporaryDirectory(prefix="nr-pusch-rt-preview-") as temporary:
                    assets = persist_scene_bundle(
                        bundle,
                        Path(temporary) / "scene",
                        source="parameterized",
                    )
                    return render_rt_scene_preview(assets, settings, view=view)
            if assets is None:
                raise ValueError("RT scene assets were not resolved")
            return render_rt_scene_preview(assets, settings, view=view)
        except (OSError, ValueError, TypeError) as exc:
            raise ApiError(400, f"RT 场景预览无效: {exc}") from exc

    @staticmethod
    def _persist_job_scene(
        directory: Path,
        assets: RtSceneAssets | None,
        bundle: ValidatedRtSceneBundle | None,
    ) -> RtSceneAssets:
        scene_root = directory / "scene"
        if bundle is not None:
            return persist_scene_bundle(bundle, scene_root, source="parameterized")
        if assets is None:
            raise ValueError("RT scene assets were not resolved")
        return copy_scene_assets(assets, scene_root)


    @staticmethod
    def _validate(kind: str, path: Path) -> None:
        with path.open("rb") as stream:
            tomllib.load(stream)
        {"tx": TxSettings, "rx": TxSettings, "channel": ChannelSettings,
         "rt": RtBeamSettings, "simulation": BlerSettings}[kind].from_toml(path)

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
        backend = prepared["channel_backend"]
        settings = prepared["settings"]
        simulation_settings: BlerSettings = settings["simulation"]
        job_id = uuid4().hex[:12]
        directory = self._job_dir(job_id)
        directory.mkdir()
        try:
            for kind, text in texts.items():
                (directory / f"{kind}.toml").write_text(text, encoding="utf-8")
            scene_assets: RtSceneAssets | None = None
            if backend == "rt":
                scene_assets = self._persist_job_scene(
                    directory,
                    prepared.get("scene_assets"),
                    prepared.get("scene_bundle"),
                )
            job: dict[str, Any] = {
                "id": job_id,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "channel_backend": backend,
                "stage": "queued",
                "stage_started_at": None,
                "cancel_requested": False,
                "scenario_id": prepared["scenario_id"],
                "scene_id": data.get("scene_id") if backend == "rt" else None,
                "scene_file": scene_assets.scene_file if scene_assets else None,
                "scene_source": scene_assets.source if scene_assets else None,
                "scene_bundle_sha256": scene_assets.bundle_sha256 if scene_assets else None,
                "user_names": (
                    [user.name for user in settings["rt"].users] if backend == "rt" else []
                ),
                "validation_summary": None,
                "snapshot_cache_key": None,
                "snapshot_cache_hit": None,
                "output_archive": None,
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
                self._finish_job(job_id, "failed", f"任务调度失败: {exc}")
            finally:
                self._pending.task_done()

    def _set_stage(self, job_id: str, stage: str, *, at: str | None = None) -> bool:
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] != "running" or job.get("cancel_requested"):
                return False
            if job.get("stage") != stage:
                job["stage"] = stage
                job["stage_started_at"] = at or _now()
                self._write_job(job)
            return True

    def _finish_job(self, job_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.get("cancel_requested"):
                status, error = "cancelled", None
            if job["status"] not in {"queued", "running"}:
                return
            job["status"] = status
            job["stage"] = {
                "completed": "done", "failed": "failed", "cancelled": "cancelled"
            }[status]
            job["stage_started_at"] = _now()
            job["error"] = error
            job["finished_at"] = _now()
            self._write_job(job)

    def _consume_stage_events(
        self, job_id: str, path: Path, offset: int
    ) -> int:
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
        except FileNotFoundError:
            return offset
        complete_bytes = chunk.rfind(b"\n") + 1
        if complete_bytes <= 0:
            return offset
        for line in chunk[:complete_bytes].splitlines():
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if (
                isinstance(event, dict)
                and event.get("type") == "stage"
                and event.get("stage") in {"tracing", "validating"}
                and isinstance(event.get("at"), str)
            ):
                self._set_stage(job_id, event["stage"], at=event["at"])
        return offset + complete_bytes

    @staticmethod
    def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                process.wait()
    def _run_child(
        self,
        job_id: str,
        command: list[str],
        *,
        timeout_s: float | None = None,
        stage_jsonl: Path | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] != "running" or job.get("cancel_requested"):
                return {"returncode": None, "cancelled": True, "timed_out": False}
        directory = self._job_dir(job_id)
        log_path = directory / "run.log"
        process: subprocess.Popen[bytes] | None = None
        try:
            with log_path.open("ab") as log:
                log.write(("$ " + " ".join(command) + "\n").encode("utf-8", errors="replace"))
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                with self._lock:
                    self._processes[job_id] = process
                    cancelled_before_registration = self._jobs[job_id].get("cancel_requested", False)
                if cancelled_before_registration and process.poll() is None:
                    try:
                        process.terminate()
                    except OSError:
                        pass
                    threading.Timer(5, self._kill_if_running, args=(process,)).start()

                event_offset = 0
                deadline = None if timeout_s is None else time.monotonic() + timeout_s
                timed_out = False
                while True:
                    if stage_jsonl is not None:
                        event_offset = self._consume_stage_events(
                            job_id, stage_jsonl, event_offset
                        )
                    returncode = process.poll()
                    if returncode is not None:
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        self._terminate_and_reap(process)
                        break
                    time.sleep(0.1)
                returncode = process.wait()
                if stage_jsonl is not None:
                    self._consume_stage_events(job_id, stage_jsonl, event_offset)
        except Exception as exc:
            if process is not None:
                self._terminate_and_reap(process)
            return {
                "returncode": None,
                "cancelled": self._cancel_requested(job_id),
                "timed_out": False,
                "error": str(exc),
            }
        finally:
            if process is not None:
                with self._lock:
                    if self._processes.get(job_id) is process:
                        self._processes.pop(job_id, None)
        return {
            "returncode": returncode,
            "cancelled": self._cancel_requested(job_id),
            "timed_out": timed_out,
        }

    def _cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return job is None or job.get("cancel_requested", False)

    @staticmethod
    def _validation_summary(report: dict[str, Any]) -> dict[str, Any]:
        rms_check = next(
            (
                check for check in report.get("checks", [])
                if isinstance(check, dict)
                and check.get("check") == "native_ofdm_fd_td_web_approximation"
            ),
            {},
        )
        return {
            "policy": report.get("policy"),
            "passed": report.get("passed"),
            "strict_fd_td_passed": report.get("strict_fd_td_passed"),
            "web_rms": rms_check.get("actual"),
            "web_rms_threshold": rms_check.get("threshold", 0.001),
            "warnings": report.get("warnings", []),
            "snapshot_hash": report.get("snapshot_hash"),
            "scene_bundle_sha256": report.get("scene_bundle_sha256"),
        }

    def _store_validation_report(self, job_id: str) -> dict[str, Any] | None:
        path = self._job_dir(job_id) / "validation.json"
        if not path.is_file():
            return None
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("RT validation.json 必须是 JSON 对象")
        with self._lock:
            job = self._jobs[job_id]
            job["validation_summary"] = self._validation_summary(report)
            self._write_job(job)
        return report

    def _verify_prepared_rt(self, job_id: str) -> dict[str, Any]:
        directory = self._job_dir(job_id)
        for name in (
            "channel_snapshot.npz",
            "channel_snapshot.json",
            "validation.json",
            "snapshot_cache.json",
        ):
            if not (directory / name).is_file():
                raise ValueError(f"RT 场景准备缺少 {name}")
        report = self._store_validation_report(job_id)
        if report is None:
            raise ValueError("RT 场景准备缺少 validation.json")
        snapshot_metadata = json.loads(
            (directory / "channel_snapshot.json").read_text(encoding="utf-8")
        )
        if not isinstance(snapshot_metadata, dict):
            raise ValueError("RT channel_snapshot.json 必须是 JSON 对象")
        cache_info = json.loads(
            (directory / "snapshot_cache.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(cache_info, dict)
            or cache_info.get("format_version") != 1
            or not isinstance(cache_info.get("cache_key"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", cache_info["cache_key"])
            or not isinstance(cache_info.get("hit"), bool)
        ):
            raise ValueError("RT snapshot cache 状态文件无效")
        with self._lock:
            job = self._jobs[job_id]
            expected_bundle = job.get("scene_bundle_sha256")
            expected_file = job.get("scene_file")
            expected_source = job.get("scene_source")
            job["snapshot_cache_key"] = cache_info["cache_key"]
            job["snapshot_cache_hit"] = cache_info["hit"]
            self._write_job(job)
        snapshot_hash = snapshot_metadata.get("array_sha256")
        if (
            report.get("format_version") != 1
            or report.get("policy") != "web-frequency-v1"
            or report.get("passed") is not True
            or report.get("snapshot_hash") != snapshot_hash
            or report.get("scene_bundle_sha256") != expected_bundle
            or snapshot_metadata.get("format_version") != 2
            or snapshot_metadata.get("scene_bundle_sha256") != expected_bundle
            or snapshot_metadata.get("scene_file") != expected_file
            or snapshot_metadata.get("scene_source") != expected_source
            or not isinstance(snapshot_hash, str)
        ):
            raise ValueError(
                "RT 场景准入报告失败，或 snapshot/scene bundle hash 与任务资源不匹配"
            )
        return report

    def _persist_completed_rt_snapshot(self, job_id: str) -> None:
        from .rt_channel import load_rt_beam_snapshot, store_rt_beam_snapshot_cache

        directory = self._job_dir(job_id)
        if directory.is_symlink() or directory.resolve(strict=True).parent != self.runs_dir:
            raise ValueError("RT job directory escaped the configured runs directory")
        with self._lock:
            job = dict(self._jobs[job_id])
        tx_settings = TxSettings.from_toml(directory / "tx.toml")
        rt_settings = RtBeamSettings.from_toml(directory / "rt.toml")
        assets = resolve_scene_assets(
            directory / "scene",
            job["scene_file"],
            source=job["scene_source"],
        )
        snapshot = load_rt_beam_snapshot(
            directory / "channel_snapshot.npz",
            tx_settings,
            scene_root=assets.root,
        )
        cache_key = store_rt_beam_snapshot_cache(
            snapshot,
            tx_settings,
            rt_settings,
            scene_assets=assets,
            cache_dir=self.outputs_dir / "rt_snapshots",
        )
        if cache_key != job.get("snapshot_cache_key"):
            raise ValueError("RT snapshot cache key 与预检结果不一致")
        archive = self._archive_completed_rt_job(job_id)
        with self._lock:
            current = self._jobs[job_id]
            current["output_archive"] = str(Path("outputs") / "rt_runs" / job_id)
            self._write_job(current)

    def _archive_completed_rt_job(self, job_id: str) -> Path:
        source = self._job_dir(job_id)
        outputs = self.outputs_dir
        if outputs.is_symlink():
            raise ValueError("项目 outputs 目录不得是符号链接")
        outputs.mkdir(parents=True, exist_ok=True)
        if not outputs.is_dir():
            raise ValueError("项目 outputs 路径必须是目录")
        archive_root = outputs / "rt_runs"
        if archive_root.is_symlink():
            raise ValueError("RT 结果目录不得是符号链接")
        archive_root.mkdir(mode=0o700, exist_ok=True)
        if archive_root.resolve(strict=True).parent != outputs.resolve(strict=True):
            raise ValueError("RT 结果目录必须直接位于 outputs 下")
        os.chmod(archive_root, 0o700)
        destination = archive_root / job_id
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"RT 结果归档已存在: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{job_id}-", dir=archive_root))
        try:
            for current, directories, filenames in os.walk(source, followlinks=False):
                source_directory = Path(current)
                relative_directory = source_directory.relative_to(source)
                target_directory = temporary / relative_directory
                target_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                for name in directories:
                    child = source_directory / name
                    if child.is_symlink() or not child.is_dir():
                        raise ValueError(f"RT 结果包含非目录资源: {child}")
                    (target_directory / name).mkdir(mode=0o700)
                for name in filenames:
                    child = source_directory / name
                    if child.is_symlink():
                        raise ValueError(f"RT 结果包含符号链接: {child}")
                    if relative_directory == Path(".") and name == "job.json":
                        continue
                    if not child.is_file():
                        raise ValueError(f"RT 结果包含非普通文件: {child}")
                    shutil.copyfile(child, target_directory / name)
            with self._lock:
                archived_job = dict(self._jobs[job_id])
            archived_job.update(
                status="completed",
                stage="done",
                finished_at=_now(),
                error=None,
                cancel_requested=False,
                output_archive=str(Path("outputs") / "rt_runs" / job_id),
            )
            (temporary / "job.json").write_text(
                json.dumps(archived_job, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.rename(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def _execute(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "queued":
                return
            job["status"] = "running"
            job["started_at"] = _now()
            backend = job.get("channel_backend", "cdl")
            initial_stage = "tracing" if backend == "rt" else "simulating"
            job["stage"] = initial_stage
            job["stage_started_at"] = _now()
            self._write_job(job)
        directory = self._job_dir(job_id)
        try:
            if backend == "rt":
                stage_jsonl = directory / "stage.jsonl"
                stage_jsonl.write_text("", encoding="utf-8")
                prepare_command = [
                    self.python, "-u", "-m", "nr_pusch.cli.web_rt_prepare",
                    "--tx-config", str(directory / "tx.toml"),
                    "--rt-config", str(directory / "rt.toml"),
                    "--simulation-config", str(directory / "simulation.toml"),
                    "--scene-root", str(directory / "scene"),
                    "--output", str(directory / "channel_snapshot.npz"),
                    "--validation-output", str(directory / "validation.json"),
                    "--stage-jsonl", str(stage_jsonl),
                    "--cache-dir", str(self.outputs_dir / "rt_snapshots"),
                ]
                prepared = self._run_child(
                    job_id, prepare_command, timeout_s=600, stage_jsonl=stage_jsonl
                )
                if prepared["cancelled"]:
                    self._finish_job(job_id, "cancelled")
                    return
                if prepared["timed_out"]:
                    self._finish_job(
                        job_id,
                        "failed",
                        "RT 场景准备超过 600 秒，已停止；请减少场景或追踪预算",
                    )
                    return
                if prepared.get("error") is not None:
                    self._finish_job(
                        job_id, "failed", f"无法启动 RT 场景准备进程: {prepared['error']}"
                    )
                    return
                if prepared["returncode"] != 0:
                    try:
                        self._store_validation_report(job_id)
                    except (OSError, ValueError, json.JSONDecodeError):
                        pass
                    self._finish_job(
                        job_id,
                        "failed",
                        f"RT 场景追踪或准入失败，退出码 {prepared['returncode']}；请查看 validation.json 与运行日志。",
                    )
                    return
                try:
                    self._verify_prepared_rt(job_id)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self._finish_job(job_id, "failed", str(exc))
                    return
                if self._cancel_requested(job_id):
                    self._finish_job(job_id, "cancelled")
                    return
                if not self._set_stage(job_id, "simulating"):
                    self._finish_job(job_id, "cancelled")
                    return
                command = [
                    self.python, "-u", "-m", "nr_pusch.cli.beam_bler",
                    "--tx-config", str(directory / "tx.toml"),
                    "--channel-snapshot", str(directory / "channel_snapshot.npz"),
                    "--scene-root", str(directory / "scene"),
                    "--simulation-config", str(directory / "simulation.toml"),
                    "--web-validation", str(directory / "validation.json"),
                    "--output", str(directory / "results.csv"),
                    "--progress-jsonl", str(directory / "progress.jsonl"),
                ]
            else:
                if not self._set_stage(job_id, "simulating"):
                    self._finish_job(job_id, "cancelled")
                    return
                command = [
                    self.python, "-u", "-m", "nr_pusch.cli.bler",
                    "--tx-config", str(directory / "tx.toml"),
                    "--channel-config", str(directory / "channel.toml"),
                    "--simulation-config", str(directory / "simulation.toml"),
                    "--output", str(directory / "results.csv"),
                    "--progress-jsonl", str(directory / "progress.jsonl"),
                    "--prior-dir", str(self.prior_dir),
                ]
            simulated = self._run_child(job_id, command)
            if simulated["cancelled"]:
                self._finish_job(job_id, "cancelled")
                return
            if simulated.get("error") is not None:
                self._finish_job(
                    job_id, "failed", f"无法启动仿真进程: {simulated['error']}"
                )
                return
            if simulated["returncode"] != 0:
                self._finish_job(
                    job_id,
                    "failed",
                    f"仿真进程退出码 {simulated['returncode']}；请查看运行日志。",
                )
                return
            missing = [
                name for name in ("results.csv", "results.json")
                if not (directory / name).is_file()
            ]
            if missing:
                self._finish_job(
                    job_id, "failed", f"仿真输出缺少 {', '.join(missing)}；未标记为完成。"
                )
                return
            if backend == "rt":
                if self._cancel_requested(job_id):
                    self._finish_job(job_id, "cancelled")
                    return
                self._persist_completed_rt_snapshot(job_id)
            self._finish_job(job_id, "completed")
        except Exception as exc:
            traceback.print_exc()
            self._finish_job(job_id, "failed", f"任务执行失败: {exc}")

    def cancel_run(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApiError(404, "运行不存在")
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["stage"] = "cancelled"
                job["stage_started_at"] = _now()
                job["finished_at"] = _now()
            elif job["status"] == "running":
                job["cancel_requested"] = True
                process = self._processes.get(job_id)
                if process is not None and process.poll() is None:
                    try:
                        process.terminate()
                    except OSError:
                        pass
                    threading.Timer(5, self._kill_if_running, args=(process,)).start()
            self._write_job(job)
        return self.get_run(job_id)

    @staticmethod
    def _kill_if_running(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

    @staticmethod
    def _rt_point_group(
        rows: Any, user_names: list[str]
    ) -> tuple[tuple[str, str, float], list[dict[str, Any]]] | None:
        if not isinstance(rows, list) or len(rows) != 5 or len(user_names) != 4:
            return None
        by_user: dict[str, dict[str, Any]] = {}
        common: tuple[str, str, float] | None = None
        valid_statuses = {
            "complete", "infeasible_rank", "singular_noise_covariance", "skipped",
        }
        for source in rows:
            if not isinstance(source, dict):
                return None
            user = source.get("user")
            estimator, detector = source.get("channel_estimator"), source.get("detector")
            snr = source.get("snr_db")
            if (
                not isinstance(user, str)
                or user not in {*user_names, "all"}
                or user in by_user
                or not isinstance(estimator, str)
                or not estimator
                or not isinstance(detector, str)
                or not detector
                or isinstance(snr, bool)
            ):
                return None
            try:
                snr_db = float(snr)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(snr_db) or source.get("status") not in valid_statuses:
                return None
            if source.get("status") == "skipped" and (
                source.get("frames") != 0
                or source.get("transport_blocks") != 0
                or source.get("block_errors") != 0
                or not isinstance(source.get("reason"), str)
                or any(
                    source.get(field) is not None
                    for field in (
                        "bler", "bler_ci95_low", "bler_ci95_high", "crc_fail_rate",
                        "ber", "actual_snr_db",
                    )
                )
            ):
                return None
            key = (estimator, detector, snr_db)
            if common is None:
                common = key
            elif key != common:
                return None
            by_user[user] = {**source, "snr_db": snr_db}
        if set(by_user) != {*user_names, "all"}:
            return None
        statuses = {row.get("status") for row in by_user.values()}
        if len(statuses) != 1:
            return None
        return common, [by_user[name] for name in (*user_names, "all")]

    @staticmethod
    def _rt_csv_row(raw: dict[str, str]) -> dict[str, Any]:
        integer_fields = {
            "seed", "frames", "transport_blocks", "block_errors", "crc_failures",
            "bit_errors", "bits",
        }
        float_fields = {
            "snr_db", "post_combiner_ratio", "bler", "bler_ci95_low", "bler_ci95_high",
            "crc_fail_rate", "ber", "runtime_s", "reference_snr_db", "actual_snr_db",
        }
        row: dict[str, Any] = dict(raw)
        for key in integer_fields | float_fields:
            value = raw.get(key)
            if value is None or value.strip() == "" or value.strip().lower() == "null":
                row[key] = None
            else:
                try:
                    row[key] = int(value) if key in integer_fields else float(value)
                except ValueError:
                    row[key] = None
        for key in ("reason",):
            if row.get(key) == "":
                row[key] = None
        return row

    def _rt_csv_points(
        self, path: Path, user_names: list[str]
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, float], dict[str, dict[str, Any]]] = {}
        with path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                row = self._rt_csv_row(raw)
                estimator, detector, snr = (
                    row.get("channel_estimator"), row.get("detector"), row.get("snr_db")
                )
                if not isinstance(estimator, str) or not isinstance(detector, str):
                    continue
                if isinstance(snr, bool) or not isinstance(snr, (int, float)) or not math.isfinite(float(snr)):
                    continue
                key = (estimator, detector, float(snr))
                user = row.get("user")
                if isinstance(user, str) and user in {*user_names, "all"}:
                    grouped.setdefault(key, {})[user] = row
        points: list[dict[str, Any]] = []
        for rows in grouped.values():
            group = self._rt_point_group(
                [rows[name] for name in (*user_names, "all") if name in rows],
                user_names,
            )
            if group is not None:
                points.extend(group[1])
        return points

    def _rt_points(self, directory: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
        user_names = list(job.get("user_names") or [])
        progress = directory / "progress.jsonl"
        groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
        if progress.is_file():
            data = progress.read_bytes()
            for line in data.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    continue
                try:
                    event = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(event, dict) or event.get("type") != "point":
                    continue
                group = self._rt_point_group(event.get("rows"), user_names)
                if group is not None:
                    groups[group[0]] = group[1]
        if not groups and (directory / "results.csv").is_file():
            return self._rt_csv_points(directory / "results.csv", user_names)
        return [row for rows in groups.values() for row in rows]

    def _points(self, job_id: str) -> list[dict[str, Any]]:
        directory = self._job_dir(job_id)
        with self._lock:
            job = dict(self._jobs[job_id])
        if job.get("channel_backend", "cdl") == "rt":
            return self._rt_points(directory, job)
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
            should_load_validation = (
                job.get("channel_backend") == "rt"
                and not job.get("validation_summary")
                and (self._job_dir(job_id) / "validation.json").is_file()
            )
        if should_load_validation:
            try:
                self._store_validation_report(job_id)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        with self._lock:
            result = dict(self._jobs[job_id])
        directory = self._job_dir(job_id)
        log = directory / "run.log"
        points = self._points(job_id)
        result["points"] = points
        result["logs"] = log.read_text(encoding="utf-8", errors="replace").splitlines()[-100:] if log.is_file() else []
        result["output_csv"] = f"/api/runs/{job_id}/results.csv" if (directory / "results.csv").is_file() else None
        result["output_manifest"] = f"/api/runs/{job_id}/results.json" if (directory / "results.json").is_file() else None
        if result.get("channel_backend") == "rt":
            result["completed_points"] = sum(row.get("user") == "all" for row in points)
            result["output_validation"] = (
                f"/api/runs/{job_id}/validation.json"
                if (directory / "validation.json").is_file() else None
            )
            result["validation_url"] = result["output_validation"]
            result["output_snapshot_npz"] = (
                f"/api/runs/{job_id}/channel_snapshot.npz"
                if (directory / "channel_snapshot.npz").is_file() else None
            )
            result["output_snapshot_json"] = (
                f"/api/runs/{job_id}/channel_snapshot.json"
                if (directory / "channel_snapshot.json").is_file() else None
            )
            result["output_reproducibility"] = (
                f"/api/runs/{job_id}/reproducibility.zip"
                if all((directory / name).is_file() for name in ("tx.toml", "rt.toml", "simulation.toml"))
                and (directory / "scene").is_dir()
                else None
            )
            snapshot_json = directory / "channel_snapshot.json"
            if snapshot_json.is_file():
                try:
                    metadata = json.loads(snapshot_json.read_text(encoding="utf-8"))
                    if isinstance(metadata, dict):
                        result["trace_variant"] = metadata.get("rt_variant")
                except (OSError, ValueError, TypeError):
                    pass
            manifest = directory / "results.json"
            if manifest.is_file():
                try:
                    metadata = json.loads(manifest.read_text(encoding="utf-8"))
                    if isinstance(metadata, dict):
                        result["device"] = metadata.get("device")
                except (OSError, ValueError, TypeError):
                    pass
        return result

    def list_runs(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            ids = sorted(self._jobs, key=lambda key: self._jobs[key]["created_at"], reverse=True)
        return {"runs": [self.get_run(job_id) for job_id in ids]}

    def delete_runs(self, job_ids: Any) -> dict[str, Any]:
        """Delete terminal run records and their per-job result artifacts."""
        if not isinstance(job_ids, list) or not job_ids:
            raise ApiError(400, "ids 必须是非空运行 ID 数组")
        if any(not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id) for job_id in job_ids):
            raise ApiError(400, "ids 包含无效运行 ID")
        if len(set(job_ids)) != len(job_ids):
            raise ApiError(400, "ids 不能包含重复运行 ID")

        archive_root = self.outputs_dir / "rt_runs"
        with self._lock:
            missing = [job_id for job_id in job_ids if job_id not in self._jobs]
            if missing:
                raise ApiError(404, f"运行不存在: {', '.join(missing)}")
            active = [
                job_id for job_id in job_ids
                if self._jobs[job_id].get("status") not in {"completed", "failed", "cancelled"}
            ]
            if active:
                raise ApiError(409, f"运行尚未结束，不能删除: {', '.join(active)}")
            if self.outputs_dir.is_symlink() or archive_root.is_symlink():
                raise ApiError(409, "RT 结果归档路径无效，拒绝删除")

            paths = {
                job_id: (self._job_dir(job_id), archive_root / job_id)
                for job_id in job_ids
            }
            for job_id, job_paths in paths.items():
                for path in job_paths:
                    if path.is_symlink():
                        raise ApiError(409, f"运行 {job_id} 的数据路径是符号链接，拒绝删除")
                    if path.exists() and not path.is_dir():
                        raise ApiError(409, f"运行 {job_id} 的数据路径不是目录，拒绝删除")
            for job_id in job_ids:
                for path in paths[job_id]:
                    if path.is_dir():
                        shutil.rmtree(path)
                self._jobs.pop(job_id)
        return {"deleted_ids": job_ids, "deleted_count": len(job_ids)}


    def _make_reproducibility_zip(self, job_id: str) -> Path:
        directory = self._job_dir(job_id)
        destination = directory / "reproducibility.zip"
        with tempfile.NamedTemporaryFile(
            prefix=".reproducibility-", suffix=".zip.tmp", dir=directory, delete=False
        ) as temporary_stream:
            temporary = Path(temporary_stream.name)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in ("tx.toml", "rt.toml", "simulation.toml", "validation.json"):
                    source = directory / name
                    if source.is_file() and not source.is_symlink():
                        archive.write(source, arcname=name)
                scene_root = directory / "scene"
                if scene_root.is_dir() and not scene_root.is_symlink():
                    for source in sorted(scene_root.rglob("*")):
                        if source.is_symlink():
                            raise ValueError("job scene package contains a symbolic link")
                        if source.is_file():
                            relative = source.relative_to(scene_root).as_posix()
                            archive.write(source, arcname=f"scene/{relative}")
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def result_file(self, job_id: str, name: str) -> Path:
        if name not in {
            "results.csv", "results.json", "validation.json", "channel_snapshot.npz",
            "channel_snapshot.json", "reproducibility.zip",
        }:
            raise ApiError(404, "结果文件不存在")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not _JOB_ID.fullmatch(job_id):
                raise ApiError(404, "运行不存在")
            if name == "reproducibility.zip" and job.get("channel_backend") != "rt":
                raise ApiError(404, "结果文件不存在")
        path = (
            self._make_reproducibility_zip(job_id)
            if name == "reproducibility.zip"
            else self._job_dir(job_id) / name
        )
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

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str,
            extra_headers: tuple[tuple[str, str], ...] = (),
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in extra_headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_file(
            self,
            status: int,
            path: Path,
            content_type: str,
            *,
            download_name: str | None = None,
        ) -> None:
            size = path.stat().st_size
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download_name is not None:
                self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            self.end_headers()
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
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
                elif method == "GET" and path == "/api/rt/options":
                    self._json(200, app.rt_options())
                elif method == "POST" and path == "/api/rt/config":
                    self._json(200, app.rt_config(self._body()))
                elif method == "POST" and path == "/api/rt/scene-preview":
                    self._send(
                        200,
                        app.rt_scene_preview(self._body()),
                        "image/png",
                    )
                elif method == "GET" and path == "/api/rt/scene-template.zip":
                    self._send(
                        200,
                        make_scene_template_zip(),
                        "application/zip",
                        (("Content-Disposition", 'attachment; filename="scene-template.zip"'),),
                    )
                elif method == "POST" and path == "/api/rt/scenes":
                    self._json(
                        HTTPStatus.CREATED,
                        app.upload_scene(self._body(max_bytes=12 * 1024 * 1024)),
                    )
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
                elif path == "/api/runs" and method == "DELETE":
                    self._json(200, app.delete_runs(self._body().get("ids")))
                elif path == "/api/rx/decode" and method == "POST":
                    self._json(200, app.decode_external(self._body(max_bytes=18 * 1024 * 1024)))
                elif len(parts) == 5 and parts[:3] == ["api", "rx", "decode"] and method == "GET":
                    file = app.decode_file(parts[3], parts[4])
                    mime = "application/json; charset=utf-8" if file.suffix == ".json" else "application/octet-stream"
                    self._send_file(200, file, mime, download_name=file.name)
                elif len(parts) == 3 and parts[:2] == ["api", "runs"] and method == "GET":
                    self._json(200, app.get_run(parts[2]))
                elif len(parts) == 3 and parts[:2] == ["api", "runs"] and method == "DELETE":
                    self._json(200, app.delete_runs([parts[2]]))
                elif len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "cancel" and method == "POST":
                    self._json(200, app.cancel_run(parts[2]))
                elif len(parts) == 4 and parts[:2] == ["api", "runs"] and method == "GET":
                    file = app.result_file(parts[2], parts[3])
                    mime_types = {
                        ".csv": "text/csv; charset=utf-8",
                        ".json": "application/json; charset=utf-8",
                        ".npz": "application/octet-stream",
                        ".zip": "application/zip",
                    }
                    self._send_file(
                        200, file, mime_types.get(file.suffix, "application/octet-stream"),
                        download_name=file.name,
                    )
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

        def do_DELETE(self) -> None:
            self._handle("DELETE")

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
