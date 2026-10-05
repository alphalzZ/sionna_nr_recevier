"""Sionna RT path snapshots and static four-beam PUSCH channel adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
import torch
from sionna.phy.channel import cir_to_time_channel, subcarrier_frequencies, time_lag_discrete_time_channel
from sionna.phy.channel import ApplyTimeChannel
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

from nr_pusch.beamforming import combine_beams, make_beam_weights, user_power_scales
from nr_pusch.channel import ChannelResult, FrequencyChannelResult
from nr_pusch.config import TxSettings
from nr_pusch.device import use_device
from nr_pusch.rt_config import RtBeamSettings
from nr_pusch.rt_scene_assets import RtSceneAssets, resolve_builtin_scene_assets, resolve_scene_assets
from nr_pusch.transmitter import NrPuschTx

_FORMAT_VERSION = 2

_ARRAY_KEYS = (
    "path_a",
    "path_tau_s",
    "path_valid",
    "ta_s",
    "element_positions_m",
    "weights",
    "h_ant",
    "h_beam",
    "taps_ant",
    "taps_beam",
    "frequencies_hz",
    "path_theta_r_rad",
    "path_phi_r_rad",
    "path_vertices_m",
    "path_interactions",
)


@dataclass(frozen=True)
class RtBeamSnapshot:
    """Validated static RT channel with unpowered array and beam responses."""

    path_a: np.ndarray
    path_tau_s: np.ndarray
    path_valid: np.ndarray
    ta_s: np.ndarray
    element_positions_m: np.ndarray
    weights: np.ndarray
    h_ant: np.ndarray
    h_beam: np.ndarray
    taps_ant: np.ndarray
    taps_beam: np.ndarray
    frequencies_hz: np.ndarray
    path_theta_r_rad: np.ndarray
    path_phi_r_rad: np.ndarray
    path_vertices_m: np.ndarray
    path_interactions: np.ndarray
    metadata: dict[str, Any]

    @property
    def power_scale_db(self) -> np.ndarray:
        """Configured relative per-UE transmit power, in TX user order."""
        return np.asarray(
            [user["power_scale_db"] for user in self.metadata["rt_config"]["users"]],
            dtype=np.float64,
        )


def prepare_rt_beam_snapshot(
    tx_settings: TxSettings,
    rt_settings: RtBeamSettings,
    *,
    scene_assets: RtSceneAssets | None = None,
) -> RtBeamSnapshot:
    """Trace a configured scene and prepare its static shared-array channel."""
    tx_settings.validate()
    rt_settings.validate_transmitter(tx_settings)
    scene_assets = _resolve_scene_assets(rt_settings, scene_assets)
    phy_tx = NrPuschTx(tx_settings, device="cpu")
    resource_grid = phy_tx._tx_freq.resource_grid
    sample_rate_hz = int(resource_grid.fft_size * resource_grid.subcarrier_spacing)
    frequencies_hz = (
        subcarrier_frequencies(resource_grid.fft_size, resource_grid.subcarrier_spacing)
        .to(torch.float64)
        .cpu()
        .numpy()
    )
    tx_geometry = _tx_resource_geometry(tx_settings, resource_grid)

    scene_path = scene_assets.root / scene_assets.scene_file
    scene = load_scene(str(scene_path))
    scene.frequency = rt_settings.rt.carrier_frequency_hz
    wavelength_m = 299_792_458.0 / rt_settings.rt.carrier_frequency_hz
    scene_materials = {
        name: {
            "relative_permittivity": float(material.relative_permittivity[0]),
            "conductivity_s_per_m": float(material.conductivity[0]),
            "thickness_m": float(material.thickness[0]),
            "scattering_coefficient": float(material.scattering_coefficient[0]),
        }
        for name, material in scene.radio_materials.items()
    }
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=rt_settings.receiver.num_rows,
        num_cols=rt_settings.receiver.num_cols,
        vertical_spacing=rt_settings.receiver.vertical_spacing_wavelengths,
        horizontal_spacing=rt_settings.receiver.horizontal_spacing_wavelengths,
        pattern=rt_settings.receiver.pattern,
        polarization=rt_settings.receiver.polarization,
    )

    transmitters = [
        Transmitter(
            name=user.name,
            position=mi.Point3f(user.position_m),
            orientation=mi.Point3f(np.deg2rad(user.orientation_deg)),
            power_dbm=0.0,
        )
        for user in rt_settings.users
    ]
    receiver = Receiver(
        name=rt_settings.receiver.name,
        position=mi.Point3f(rt_settings.receiver.position_m),
        orientation=mi.Point3f(np.deg2rad(rt_settings.receiver.orientation_deg)),
    )
    scene.add([*transmitters, receiver])

    paths = PathSolver(deterministic=True)(
        scene,
        max_depth=rt_settings.rt.max_depth,
        max_num_paths_per_src=rt_settings.rt.max_num_paths_per_src,
        samples_per_src=rt_settings.rt.samples_per_src,
        synthetic_array=rt_settings.rt.synthetic_array,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        edge_diffraction=False,
        diffraction_lit_region=False,
        seed=rt_settings.rt.seed,
    )
    cir_a, cir_tau = paths.cir(
        normalize_delays=False,
        num_time_steps=1,
        out_type="numpy",
    )
    path_a, path_tau_s, path_valid = _extract_path_arrays(
        cir_a,
        cir_tau,
        _to_numpy(paths.valid),
        synthetic_array=rt_settings.rt.synthetic_array,
        num_elements=rt_settings.receiver.num_rows * rt_settings.receiver.num_cols,
        num_users=len(tx_settings.users),
    )
    path_a = np.where(path_valid, path_a, 0.0 + 0.0j)
    path_a = np.asarray(path_a, dtype=np.complex128)
    path_tau_s = np.asarray(path_tau_s, dtype=np.float64)
    path_valid = np.asarray(path_valid, dtype=np.bool_)
    if not np.all(np.isfinite(path_a)) or not np.all(np.isfinite(path_tau_s)):
        raise ValueError("RT CIR 含非有限路径系数或时延")

    valid_per_user = np.any(path_valid, axis=(0, 2))
    if not np.all(valid_per_user):
        missing = [tx_settings.users[index].name for index in np.flatnonzero(~valid_per_user)]
        locations = {
            tx_settings.users[index].name: list(rt_settings.users[index].position_m)
            for index in np.flatnonzero(~valid_per_user)
        }
        raise ValueError(f"RT 场景未找到有效路径：users={missing}, positions_m={locations}")
    ta_s = np.min(np.where(path_valid, path_tau_s, np.inf), axis=(0, 2))
    tau_res_s = np.where(path_valid, path_tau_s - ta_s[None, :, None], 0.0)
    if np.any(tau_res_s[path_valid] < -1e-12):
        raise ValueError("RT 公共 TA 产生负残余路径时延")
    tau_res_s = np.maximum(tau_res_s, 0.0)

    l_min = rt_settings.receiver.l_min
    l_max = time_lag_discrete_time_channel(
        float(sample_rate_hz), rt_settings.receiver.max_delay_spread_s
    )[1]
    sample_delays = tau_res_s * sample_rate_hz
    path_energy = np.abs(path_a) ** 2
    total_path_energy = float(np.sum(path_energy[path_valid]))
    outside = path_valid & ((sample_delays < l_min) | (sample_delays > l_max))
    outside_path_energy = float(np.sum(path_energy[outside]))
    outside_energy_ratio = outside_path_energy / total_path_energy if total_path_energy else math.inf
    if outside_energy_ratio > 0.01:
        raise ValueError(
            "RT 有效路径超出固定 tap window 的能量超过 1%："
            f"ratio={outside_energy_ratio:.6g}, window=[{l_min},{l_max}]"
        )

    h_ant = _sum_path_frequency_response(path_a, tau_res_s, path_valid, frequencies_hz)
    array_positions = scene.rx_array.rotate(wavelength_m, receiver.orientation)
    element_positions_m = np.stack(
        (
            _to_numpy(array_positions.x),
            _to_numpy(array_positions.y),
            _to_numpy(array_positions.z),
        ),
        axis=1,
    ).astype(np.float64, copy=False)
    steering_directions, steering_angles = _steering_directions(rt_settings)
    weights = make_beam_weights(
        element_positions_m,
        steering_directions,
        rt_settings.rt.carrier_frequency_hz,
        phase_bits=rt_settings.beams.phase_bits,
    )
    h_beam = combine_beams(h_ant, weights)

    taps_input_a = torch.as_tensor(path_a, dtype=torch.complex128)
    taps_input_tau = torch.as_tensor(tau_res_s, dtype=torch.float64)
    taps_input_a = taps_input_a[None, None, :, :, None, :, None]
    taps_input_tau = taps_input_tau[None, None, :, :, None, :]
    taps_all = cir_to_time_channel(
        float(sample_rate_hz),
        taps_input_a,
        taps_input_tau,
        l_min,
        l_max,
        normalize=False,
    )
    taps_ant = taps_all[0, 0, :, :, 0, 0, :].cpu().numpy()
    taps_beam = combine_beams(taps_ant, weights)
    cp_spans, cp_intervals, cp_outside_energy, cp_sufficient = _cp_energy_diagnostics(
        taps_beam,
        l_min=l_min,
        cyclic_prefix_length=int(resource_grid.cyclic_prefix_length),
    )

    cfr_raw = paths.cfr(
        mi.Float(frequencies_hz.tolist()),
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    )
    cfr_ant = _extract_cfr_array(
        cfr_raw,
        num_elements=path_a.shape[0],
        num_users=path_a.shape[1],
        num_subcarriers=frequencies_hz.size,
    )
    cfr_residual = cfr_ant * np.exp(
        2j * np.pi * frequencies_hz[None, None, :] * ta_s[None, :, None]
    )
    cfr_relative_error = _relative_error(cfr_residual, h_ant)
    if cfr_relative_error > 1e-5:
        raise ValueError(
            "RT CIR 路径和与 CFR 在公共 TA 坐标下不一致："
            f"relative_error={cfr_relative_error:.6g}"
        )

    versions = _runtime_versions()
    metadata: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "versions": versions,
        "rt_variant": mi.variant(),
        "rt_config": rt_settings.to_dict(),
        "tx_resource_geometry": tx_geometry,
        "tx_resource_geometry_sha256": _json_sha256(tx_geometry),
        "config_sha256": _json_sha256({
            "rt_config": rt_settings.to_dict(),
            "tx_resource_geometry": tx_geometry,
            "scene_bundle_sha256": scene_assets.bundle_sha256,
        }),
        "scene": rt_settings.rt.scene,
        "scene_source": scene_assets.source,
        "scene_file": scene_assets.scene_file,
        "scene_bundle_sha256": scene_assets.bundle_sha256,
        "scene_materials": scene_materials,
        "scene_asset_sha256": scene_assets.file_sha256,
        "users": [user.name for user in tx_settings.users],
        "tx_mcs": {
            "table": tx_settings.pusch.mcs_table,
            "index": tx_settings.pusch.mcs_index,
        },
        "solver": {
            "deterministic": True,
            "seed": rt_settings.rt.seed,
            "max_depth": rt_settings.rt.max_depth,
            "samples_per_src": rt_settings.rt.samples_per_src,
            "max_num_paths_per_src": rt_settings.rt.max_num_paths_per_src,
            "synthetic_array": rt_settings.rt.synthetic_array,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": False,
            "diffraction": False,
            "edge_diffraction": False,
        },
        "carrier_frequency_hz": rt_settings.rt.carrier_frequency_hz,
        "receiver_position_m": list(rt_settings.receiver.position_m),
        "receiver_orientation_deg": list(rt_settings.receiver.orientation_deg),
        "array_position_reference": "receiver_phase_center; coordinates rotated into world axes",
        "element_spacing_m": {
            "horizontal": rt_settings.receiver.horizontal_spacing_wavelengths * wavelength_m,
            "vertical": rt_settings.receiver.vertical_spacing_wavelengths * wavelength_m,
        },
        "steering_directions_world": steering_directions.tolist(),
        "steering_angles_az_el_rad": steering_angles.tolist(),
        "sample_rate_hz": sample_rate_hz,
        "fft_size": int(resource_grid.fft_size),
        "num_ofdm_symbols": int(resource_grid.num_ofdm_symbols),
        "subcarrier_spacing_hz": float(resource_grid.subcarrier_spacing),
        "cyclic_prefix_length_samples": int(resource_grid.cyclic_prefix_length),
        "l_min": l_min,
        "l_max": l_max,
        "max_delay_spread_s": rt_settings.receiver.max_delay_spread_s,
        "tap_out_of_window_path_energy_ratio": outside_energy_ratio,
        "path_cfr_relative_error_after_common_ta": cfr_relative_error,
        "cp_energy_interval_samples_by_beam_user": cp_spans,
        "cp_energy_interval_bounds_samples_by_beam_user": cp_intervals,
        "cp_outside_energy_ratio_by_beam_user": cp_outside_energy,
        "cp_outside_energy_ratio_max": max(max(row) for row in cp_outside_energy),
        "cp_sufficient": cp_sufficient,
        "frequency_axis": "baseband_offset_hz",
        "ta_semantics": "minimum valid absolute path delay across all RX elements per UE",
        "path_coefficient_semantics": "CIR coefficient includes carrier propagation phase; not power-scaled",
        "transmitter_power_dbm_for_rt_metadata_only": 0.0,
        "power_scale_db": [user.power_scale_db for user in rt_settings.users],
        "axis_order": {
            "path_a": ["element", "user", "path"],
            "path_tau_s": ["element", "user", "path"],
            "path_valid": ["element", "user", "path"],
            "ta_s": ["user"],
            "element_positions_m": ["element", "xyz"],
            "weights": ["element", "beam"],
            "h_ant": ["element", "user", "fft_bin"],
            "h_beam": ["beam", "user", "fft_bin"],
            "taps_ant": ["element", "user", "tap"],
            "taps_beam": ["beam", "user", "tap"],
            "frequencies_hz": ["fft_bin"],
            "path_theta_r_rad": ["element", "user", "path"],
            "path_phi_r_rad": ["element", "user", "path"],
            "path_vertices_m": _diagnostic_axis_order(
                paths.vertices, rt_settings.rt.synthetic_array, vertices=True
            ),
            "path_interactions": _diagnostic_axis_order(
                paths.interactions, rt_settings.rt.synthetic_array, vertices=False
            ),
        },
        "valid_path_count_per_user": np.sum(np.any(path_valid, axis=0), axis=1).astype(int).tolist(),
    }
    snapshot = RtBeamSnapshot(
        path_a=path_a,
        path_tau_s=path_tau_s,
        path_valid=path_valid,
        ta_s=ta_s,
        element_positions_m=element_positions_m,
        weights=weights,
        h_ant=h_ant,
        h_beam=h_beam,
        taps_ant=taps_ant,
        taps_beam=taps_beam,
        frequencies_hz=frequencies_hz,
        path_theta_r_rad=_expand_path_diagnostic(
            _to_numpy(paths.theta_r), rt_settings.rt.synthetic_array, path_a.shape
        ),
        path_phi_r_rad=_expand_path_diagnostic(
            _to_numpy(paths.phi_r), rt_settings.rt.synthetic_array, path_a.shape
        ),
        path_vertices_m=_to_numpy(paths.vertices),
        path_interactions=_to_numpy(paths.interactions),
        metadata=metadata,
    )
    _validate_snapshot(snapshot, tx_settings, tx_geometry)
    return snapshot


def save_rt_beam_snapshot(
    snapshot: RtBeamSnapshot,
    output: str | Path,
) -> tuple[Path, Path]:
    """Write a compressed numeric snapshot and its JSON manifest."""
    output_path = Path(output)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("RT channel 工件路径必须以 .npz 结尾")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _snapshot_arrays(snapshot)
    metadata = dict(snapshot.metadata)
    metadata["array_sha256"] = _arrays_sha256(arrays)
    with output_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, json_path


def load_rt_beam_snapshot(
    path: str | Path,
    tx_settings: TxSettings,
    *,
    scene_root: str | Path | None = None,
) -> RtBeamSnapshot:
    """Load a snapshot only when its PHY geometry, scene identity, versions, and arrays match."""
    tx_settings.validate()
    tx_geometry, resource_grid = _read_tx_resource_geometry(tx_settings)
    npz_path = Path(path)
    json_path = npz_path.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"RT snapshot NPZ/JSON sidecar 缺失：{npz_path}, {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != _FORMAT_VERSION:
        raise ValueError("RT snapshot format_version 不匹配；请用当前版本重新 prepare")
    if metadata.get("versions") != _runtime_versions():
        raise ValueError("RT snapshot 软件版本与当前运行环境不匹配")
    rt_config_raw = metadata.get("rt_config")
    if not isinstance(rt_config_raw, dict) or not isinstance(rt_config_raw.get("rt"), dict):
        raise ValueError("RT snapshot 缺少有效 RT 配置")
    rt_settings = RtBeamSettings.from_dict(rt_config_raw)
    rt_settings.validate_transmitter(tx_settings)
    scene_name = rt_settings.rt.scene
    if metadata.get("scene") != scene_name:
        raise ValueError("RT snapshot scene 名称与配置不匹配")
    scene_file = metadata.get("scene_file")
    scene_source = metadata.get("scene_source")
    if not isinstance(scene_file, str) or not isinstance(scene_source, str):
        raise ValueError("RT snapshot 缺少 scene_file 或 scene_source")
    if scene_root is None:
        if scene_source != "builtin":
            raise ValueError("RT snapshot 需要原始 scene_root 才能验证资产；请提供 --scene-root")
        assets = _resolve_scene_assets(rt_settings, None)
    else:
        assets = _resolve_scene_assets(
            rt_settings,
            resolve_scene_assets(scene_root, scene_file, source=scene_source),
        )
    if assets.scene_file != scene_file:
        raise ValueError("RT snapshot scene_file 与当前 scene root 不匹配")
    if metadata.get("scene_asset_sha256") != assets.file_sha256:
        raise ValueError("RT snapshot 场景资源 hash 与当前 scene root 不匹配")
    if metadata.get("scene_bundle_sha256") != assets.bundle_sha256:
        raise ValueError("RT snapshot scene bundle hash 与当前 scene root 不匹配")
    if metadata.get("frequency_axis") != "baseband_offset_hz":
        raise ValueError("RT snapshot frequency_axis 必须为 baseband_offset_hz")
    if metadata.get("tx_resource_geometry") != tx_geometry:
        raise ValueError("RT snapshot TX resource-grid/DMRS geometry 与当前 TX 配置不匹配")
    if metadata.get("tx_resource_geometry_sha256") != _json_sha256(tx_geometry):
        raise ValueError("RT snapshot TX resource geometry 摘要损坏")
    config_sha = _json_sha256(
        {
            "rt_config": rt_config_raw,
            "tx_resource_geometry": tx_geometry,
            "scene_bundle_sha256": assets.bundle_sha256,
        }
    )
    if metadata.get("config_sha256") != config_sha:
        raise ValueError("RT snapshot 配置摘要不匹配")
    expected_users = [user.name for user in tx_settings.users]
    if metadata.get("users") != expected_users:
        raise ValueError("RT snapshot UE 顺序与当前 TX 配置不匹配")

    with np.load(npz_path, allow_pickle=False) as stored:
        missing = set(_ARRAY_KEYS) - set(stored.files)
        if missing:
            raise ValueError(f"RT snapshot 缺少数组：{sorted(missing)}")
        arrays = {key: np.array(stored[key], copy=True) for key in _ARRAY_KEYS}
    if metadata.get("array_sha256") != _arrays_sha256(arrays):
        raise ValueError("RT snapshot NPZ 数组摘要校验失败")
    snapshot = RtBeamSnapshot(**arrays, metadata=metadata)
    _validate_snapshot(snapshot, tx_settings, tx_geometry, resource_grid=resource_grid)
    return snapshot


class NrPuschRtBeamChannel:
    """Apply one static RT snapshot to multi-user PUSCH grids or time IQ."""

    def __init__(self, snapshot: RtBeamSnapshot, *, device: str | None = None):
        self.snapshot = snapshot
        self.device = use_device(device)
        self._power_scales = torch.as_tensor(
            user_power_scales(snapshot.power_scale_db), dtype=torch.float32, device=self.device
        ).sqrt()
        self._h_beam = torch.as_tensor(snapshot.h_beam, dtype=torch.complex64, device=self.device)
        self._taps_beam = torch.as_tensor(snapshot.taps_beam, dtype=torch.complex64, device=self.device)
        self._apply_time_channel: ApplyTimeChannel | None = None
        self._time_block_size: int | None = None

    def apply_frequency(self, frequency_grid: torch.Tensor, resource_grid) -> FrequencyChannelResult:
        """Apply WᴴH to ``[batch,user,tx_ant,symbol,fft]`` without resampling RT."""
        if frequency_grid.ndim != 5 or frequency_grid.shape[1:3] != (4, 1):
            raise ValueError("RT frequency_grid 形状必须为 [batch,4 users,1 tx antenna,symbol,fft]")
        if not frequency_grid.is_complex():
            raise ValueError("frequency_grid 必须为复数张量")
        if frequency_grid.shape[-2:] != (resource_grid.num_ofdm_symbols, resource_grid.fft_size):
            raise ValueError("frequency_grid 的 symbol/fft 维度与 resource_grid 不一致")
        if resource_grid.fft_size != self.snapshot.frequencies_hz.size:
            raise ValueError("RT snapshot FFT size 与 resource_grid 不一致")
        if int(resource_grid.fft_size * resource_grid.subcarrier_spacing) != self.snapshot.metadata["sample_rate_hz"]:
            raise ValueError("RT snapshot sample rate 与 resource_grid 不一致")
        grid = frequency_grid.to(self.device)
        batch_size, _, _, num_symbols, fft_size = grid.shape
        channel = self._h_beam * self._power_scales[None, :, None]
        user_grid = grid[:, :, 0, :, :]
        contributions = (
            user_grid[:, :, None, :, :]
            * channel.permute(1, 0, 2)[None, :, :, None, :]
        )
        per_user_grid = contributions
        received = contributions.sum(dim=1).unsqueeze(1)
        csi = channel[None, None, :, :, None, None, :].expand(
            batch_size, 1, 4, 4, 1, num_symbols, fft_size
        )
        cp_sufficient = bool(self.snapshot.metadata.get("cp_sufficient", False))
        metadata = {
            "model": "Sionna RT static shared four-beam array",
            "domain": "frequency",
            "channel_sampling": "fixed_rt_snapshot",
            "grid_axes": ["batch", "num_rx", "rx_antenna", "ofdm_symbol", "fft_bin"],
            "per_user_grid_axes": ["batch", "user", "rx_antenna", "ofdm_symbol", "fft_bin"],
            "channel_axes": ["batch", "num_rx", "rx_antenna", "user", "tx_antenna", "ofdm_symbol", "fft_bin"],
            "num_users": 4,
            "num_rx_antennas": 4,
            "num_tx_antennas_per_user": 1,
            "num_ofdm_symbols": num_symbols,
            "fft_size": fft_size,
            "sample_rate_hz": self.snapshot.metadata["sample_rate_hz"],
            "carrier_frequency_hz": self.snapshot.metadata["carrier_frequency_hz"],
            "frequency_axis": "baseband_offset_hz",
            "cyclic_prefix_sufficient": cp_sufficient,
            "cyclic_prefix_assumption": "static per-RE model; validate CP energy before FD/TD equivalence claims",
            "channel_snapshot_config_sha256": self.snapshot.metadata["config_sha256"],
            "power_scale_db": self.snapshot.power_scale_db.tolist(),
            "ta_s": self.snapshot.ta_s.tolist(),
        }
        return FrequencyChannelResult(
            grid=received,
            per_user_grid=per_user_grid,
            channel_frequency_response=csi,
            metadata=metadata,
        )

    def apply(self, iq: torch.Tensor, sample_rate_hz: int) -> ChannelResult:
        """Apply static beam taps to ``[batch,user,tx_ant,sample]`` IQ."""
        if iq.ndim != 4 or iq.shape[1:3] != (4, 1):
            raise ValueError("RT IQ 形状必须为 [batch,4 users,1 tx antenna,samples]")
        if not iq.is_complex():
            raise ValueError("iq 必须为复数张量")
        if sample_rate_hz != self.snapshot.metadata["sample_rate_hz"]:
            raise ValueError("RT snapshot sample rate 与输入 IQ 不一致")
        iq = iq.to(self.device)
        batch_size, num_users, _, num_samples = iq.shape
        l_min = int(self.snapshot.metadata["l_min"])
        l_max = int(self.snapshot.metadata["l_max"])
        num_taps = l_max - l_min + 1
        num_channel_steps = num_samples + num_taps - 1
        if self._apply_time_channel is None or self._time_block_size != num_samples:
            self._apply_time_channel = ApplyTimeChannel(
                num_time_samples=num_samples,
                l_tot=num_taps,
                device=self.device,
            )
            self._time_block_size = num_samples
        scaled_taps = self._taps_beam * self._power_scales[None, :, None]
        taps_per_user = scaled_taps.permute(1, 0, 2)
        h_time = taps_per_user[None, :, :, None, None, None, :].expand(
            batch_size,
            num_users,
            4,
            1,
            1,
            num_channel_steps,
            num_taps,
        ).reshape(batch_size * num_users, 1, 4, 1, 1, num_channel_steps, num_taps)
        tx = iq.reshape(batch_size * num_users, 1, 1, num_samples)
        rx = self._apply_time_channel(tx, h_time)
        per_user_iq = rx[:, 0, :, :].reshape(batch_size, num_users, 4, -1)
        output = per_user_iq.sum(dim=1)
        channel_taps = h_time[:, 0, :, 0, 0, :, :].reshape(
            batch_size, num_users, 4, num_channel_steps, num_taps
        ).unsqueeze(3)
        metadata = {
            "model": "Sionna RT static shared four-beam array",
            "domain": "time",
            "channel_sampling": "fixed_rt_snapshot",
            "iq_axes": ["batch", "rx_antenna", "sample"],
            "per_user_iq_axes": ["batch", "user", "rx_antenna", "sample"],
            "channel_tap_axes": ["batch", "user", "rx_antenna", "tx_antenna", "channel_step", "tap"],
            "num_users": 4,
            "num_rx_antennas": 4,
            "tx_antennas_per_user": 1,
            "sample_rate_hz": sample_rate_hz,
            "time_lag_min": l_min,
            "time_lag_max": l_max,
            "channel_tail_samples": num_taps - 1,
            "channel_snapshot_config_sha256": self.snapshot.metadata["config_sha256"],
            "power_scale_db": self.snapshot.power_scale_db.tolist(),
            "ta_s": self.snapshot.ta_s.tolist(),
        }
        return ChannelResult(
            iq=output,
            per_user_iq=per_user_iq,
            channel_taps=channel_taps,
            sample_rate_hz=int(sample_rate_hz),
            metadata=metadata,
        )


def _extract_path_arrays(
    cir_a: Any,
    cir_tau: Any,
    valid: Any,
    *,
    synthetic_array: bool,
    num_elements: int,
    num_users: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = _to_numpy(cir_a)
    tau = _to_numpy(cir_tau)
    path_valid = np.asarray(valid, dtype=np.bool_)
    if a.ndim != 6 or a.shape[:4] != (1, num_elements, num_users, 1) or a.shape[5] != 1:
        raise ValueError(f"Sionna Paths.cir 系数轴不符合预期：{a.shape}")
    path_a = a[0, :, :, 0, :, 0]
    num_paths = path_a.shape[-1]
    if synthetic_array:
        if tau.shape != (1, num_users, num_paths) or path_valid.shape != tau.shape:
            raise ValueError(
                f"synthetic-array tau/valid 轴不符合预期：tau={tau.shape}, valid={path_valid.shape}"
            )
        path_tau = np.broadcast_to(tau[0][None, :, :], path_a.shape).copy()
        valid_array = np.broadcast_to(path_valid[0][None, :, :], path_a.shape).copy()
    else:
        expected = (1, num_elements, num_users, 1, num_paths)
        if tau.shape != expected or path_valid.shape != expected:
            raise ValueError(
                f"explicit-array tau/valid 轴不符合预期：tau={tau.shape}, valid={path_valid.shape}"
            )
        path_tau = tau[0, :, :, 0, :]
        valid_array = path_valid[0, :, :, 0, :]
    return path_a, path_tau, valid_array


def _extract_cfr_array(cfr: Any, *, num_elements: int, num_users: int, num_subcarriers: int) -> np.ndarray:
    values = _to_numpy(cfr)
    expected = (1, num_elements, num_users, 1, 1, num_subcarriers)
    if values.shape != expected:
        raise ValueError(f"Sionna Paths.cfr 轴不符合预期：expected={expected}, actual={values.shape}")
    return values[0, :, :, 0, 0, :]


def _sum_path_frequency_response(
    path_a: np.ndarray,
    tau_res_s: np.ndarray,
    path_valid: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    response = np.zeros(path_a.shape[:2] + (frequencies_hz.size,), dtype=np.complex128)
    for start in range(0, path_a.shape[-1], 16):
        stop = min(start + 16, path_a.shape[-1])
        coefficients = np.where(path_valid[:, :, start:stop], path_a[:, :, start:stop], 0.0)
        phase = np.exp(
            -2j
            * np.pi
            * tau_res_s[:, :, start:stop, None]
            * frequencies_hz[None, None, None, :]
        )
        response += np.sum(coefficients[:, :, :, None] * phase, axis=2)
    return response


def _steering_directions(settings: RtBeamSettings) -> tuple[np.ndarray, np.ndarray]:
    bs = np.asarray(settings.receiver.position_m, dtype=np.float64)
    users = np.asarray([user.position_m for user in settings.users], dtype=np.float64)
    displacement = users - bs[None, :]
    azimuth = np.arctan2(displacement[:, 1], displacement[:, 0]) + np.deg2rad(
        settings.beams.azimuth_offset_deg
    )
    elevation = np.arctan2(
        displacement[:, 2], np.hypot(displacement[:, 0], displacement[:, 1])
    ) + np.deg2rad(settings.beams.elevation_offset_deg)
    directions = np.column_stack(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        )
    )
    return directions, np.column_stack((azimuth, elevation))


def _expand_path_diagnostic(values: Any, synthetic_array: bool, target_shape: tuple[int, ...]) -> np.ndarray:
    array = _to_numpy(values)
    if synthetic_array:
        expected = (1, target_shape[1], target_shape[2])
        if array.shape != expected:
            raise ValueError(f"Sionna synthetic path diagnostic 轴不符合预期：{array.shape}")
        return np.broadcast_to(array[0][None, :, :], target_shape).copy()
    expected = (1, target_shape[0], target_shape[1], 1, target_shape[2])
    if array.shape != expected:
        raise ValueError(f"Sionna explicit path diagnostic 轴不符合预期：{array.shape}")
    return array[0, :, :, 0, :]


def _diagnostic_axis_order(value: Any, synthetic_array: bool, *, vertices: bool) -> list[str]:
    shape = _to_numpy(value).shape
    if synthetic_array:
        if len(shape) not in {4, 5} or shape[1] != 1:
            raise ValueError(f"Sionna synthetic diagnostic 轴不符合预期：{shape}")
        axes = ["depth", "receiver", "user", "path"]
    else:
        if len(shape) not in {6, 7}:
            raise ValueError(f"Sionna explicit diagnostic 轴不符合预期：{shape}")
        axes = ["depth", "receiver", "element", "user", "tx_antenna", "path"]
    if vertices:
        if shape[-1] != 3:
            raise ValueError(f"Sionna path vertices 缺少 xyz 轴：{shape}")
        axes.append("xyz")
    return axes


def _tx_resource_geometry(tx_settings: TxSettings, resource_grid: Any) -> dict[str, Any]:
    p = tx_settings.pusch
    return {
        "carrier": {
            "subcarrier_spacing_khz": tx_settings.carrier.subcarrier_spacing_khz,
            "cyclic_prefix": tx_settings.carrier.cyclic_prefix,
            "n_size_grid": tx_settings.carrier.n_size_grid,
            "n_start_grid": tx_settings.carrier.n_start_grid,
        },
        "pusch": {
            "waveform": p.waveform,
            "mapping_type": p.mapping_type,
            "symbol_allocation": list(p.symbol_allocation),
            "n_size_bwp": p.n_size_bwp,
            "n_start_bwp": p.n_start_bwp,
            "dmrs_config_type": p.dmrs_config_type,
            "dmrs_type_a_position": p.dmrs_type_a_position,
            "dmrs_additional_position": p.dmrs_additional_position,
            "dmrs_num_cdm_groups_without_data": p.dmrs_num_cdm_groups_without_data,
            "num_layers": p.num_layers,
            "num_antenna_ports": p.num_antenna_ports,
            "precoding": p.precoding,
            "tpmi": p.tpmi,
            "dmrs_length": p.dmrs_length,
            "dft_s_dmrs_port_order": p.dft_s_dmrs_port_order,
        },
        "users": [
            {"name": user.name, "dmrs_ports": list(user.dmrs_ports)}
            for user in tx_settings.users
        ],
        "resource_grid": {
            "fft_size": int(resource_grid.fft_size),
            "num_ofdm_symbols": int(resource_grid.num_ofdm_symbols),
            "cyclic_prefix_length": int(resource_grid.cyclic_prefix_length),
            "subcarrier_spacing_hz": float(resource_grid.subcarrier_spacing),
            "bandwidth_hz": float(resource_grid.bandwidth),
        },
    }


def _read_tx_resource_geometry(tx_settings: TxSettings) -> tuple[dict[str, Any], Any]:
    phy_tx = NrPuschTx(tx_settings, device="cpu")
    return _tx_resource_geometry(tx_settings, phy_tx._tx_freq.resource_grid), phy_tx._tx_freq.resource_grid


def _validate_snapshot(
    snapshot: RtBeamSnapshot,
    tx_settings: TxSettings,
    tx_geometry: dict[str, Any],
    *,
    resource_grid: Any | None = None,
) -> None:
    num_elements = int(snapshot.metadata["rt_config"]["receiver"]["num_rows"]) * int(
        snapshot.metadata["rt_config"]["receiver"]["num_cols"]
    )
    num_users = len(tx_settings.users)
    if snapshot.path_a.ndim != 3 or snapshot.path_a.shape[:2] != (num_elements, num_users):
        raise ValueError("RT snapshot path_a 形状不匹配")
    path_shape = snapshot.path_a.shape
    if snapshot.path_tau_s.shape != path_shape or snapshot.path_valid.shape != path_shape:
        raise ValueError("RT snapshot path coefficient/delay/valid 形状不匹配")
    expected_shapes = {
        "ta_s": (num_users,),
        "element_positions_m": (num_elements, 3),
        "weights": (num_elements, 4),
        "h_ant": (num_elements, num_users, snapshot.frequencies_hz.size),
        "h_beam": (4, num_users, snapshot.frequencies_hz.size),
        "taps_ant": (num_elements, num_users, int(snapshot.metadata["l_max"]) - int(snapshot.metadata["l_min"]) + 1),
        "taps_beam": (4, num_users, int(snapshot.metadata["l_max"]) - int(snapshot.metadata["l_min"]) + 1),
        "frequencies_hz": (int(tx_geometry["resource_grid"]["fft_size"]),),
        "path_theta_r_rad": path_shape,
        "path_phi_r_rad": path_shape,
    }
    for key, expected in expected_shapes.items():
        actual = np.shape(getattr(snapshot, key))
        if actual != expected:
            raise ValueError(f"RT snapshot {key} 形状不匹配：expected={expected}, actual={actual}")
    if snapshot.path_valid.dtype != np.bool_:
        raise ValueError("RT snapshot path_valid 必须为布尔类型")
    for key in _ARRAY_KEYS:
        array = getattr(snapshot, key)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"RT snapshot {key} 含非有限值")
    if not np.iscomplexobj(snapshot.path_a) or not np.iscomplexobj(snapshot.h_ant) or not np.iscomplexobj(snapshot.h_beam):
        raise ValueError("RT snapshot CIR/CFR 必须为复数数组")
    rt_config = snapshot.metadata["rt_config"]
    synthetic_array = bool(rt_config["rt"]["synthetic_array"])
    depth_size = max(1, int(rt_config["rt"]["max_depth"]))
    num_paths = path_shape[2]
    if synthetic_array:
        expected_vertices = (depth_size, 1, num_users, num_paths, 3)
        expected_interactions = (depth_size, 1, num_users, num_paths)
    else:
        expected_vertices = (depth_size, 1, num_elements, num_users, 1, num_paths, 3)
        expected_interactions = (depth_size, 1, num_elements, num_users, 1, num_paths)
    if snapshot.path_vertices_m.shape != expected_vertices:
        raise ValueError(
            f"RT snapshot path_vertices_m 形状不匹配：expected={expected_vertices}, "
            f"actual={snapshot.path_vertices_m.shape}"
        )
    if snapshot.path_interactions.shape != expected_interactions:
        raise ValueError(
            f"RT snapshot path_interactions 形状不匹配：expected={expected_interactions}, "
            f"actual={snapshot.path_interactions.shape}"
        )
    if snapshot.metadata.get("frequency_axis") != "baseband_offset_hz":
        raise ValueError("RT snapshot frequency axis 语义不匹配")
    if np.any(snapshot.path_a[~snapshot.path_valid] != 0.0):
        raise ValueError("RT snapshot invalid path coefficients must be zero")
    expected_ta = np.min(
        np.where(snapshot.path_valid, snapshot.path_tau_s, np.inf),
        axis=(0, 2),
    )
    if not np.allclose(snapshot.ta_s, expected_ta, rtol=0.0, atol=1e-15):
        raise ValueError("RT snapshot TA 与有效绝对路径时延不一致")
    if snapshot.metadata.get("l_min") != rt_config["receiver"]["l_min"]:
        raise ValueError("RT snapshot l_min 与 receiver 配置不一致")
    if snapshot.metadata.get("max_delay_spread_s") != rt_config["receiver"]["max_delay_spread_s"]:
        raise ValueError("RT snapshot delay window 与 receiver 配置不一致")
    if snapshot.metadata.get("users") != [user.name for user in tx_settings.users]:
        raise ValueError("RT snapshot UE 顺序与 TX 配置不匹配")
    if snapshot.metadata.get("tx_resource_geometry") != tx_geometry:
        raise ValueError("RT snapshot TX resource-grid/DMRS geometry 不匹配")
    if resource_grid is not None:
        expected_frequencies = (
            subcarrier_frequencies(resource_grid.fft_size, resource_grid.subcarrier_spacing)
            .to(torch.float64)
            .cpu()
            .numpy()
        )
        if not np.array_equal(snapshot.frequencies_hz, expected_frequencies):
            raise ValueError("RT snapshot baseband frequency axis 与当前 resource_grid 不匹配")
    if _relative_error(combine_beams(snapshot.h_ant, snapshot.weights), snapshot.h_beam) > 1e-12:
        raise ValueError("RT snapshot h_beam 与 Wᴴ h_ant 不一致")
    if _relative_error(combine_beams(snapshot.taps_ant, snapshot.weights), snapshot.taps_beam) > 1e-12:
        raise ValueError("RT snapshot taps_beam 与 Wᴴ taps_ant 不一致")


def _snapshot_arrays(snapshot: RtBeamSnapshot) -> dict[str, np.ndarray]:
    return {key: np.asarray(getattr(snapshot, key)) for key in _ARRAY_KEYS}


def _arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _resolve_scene_assets(
    rt_settings: RtBeamSettings,
    scene_assets: RtSceneAssets | None,
) -> RtSceneAssets:
    scene = rt_settings.rt.scene
    if scene_assets is None:
        if scene == "custom":
            raise ValueError("custom RT 场景必须显式提供 scene_assets")
        if rt_settings.geometry is not None:
            raise ValueError("参数化 RT 场景必须显式提供生成后的 scene_assets")
        return resolve_builtin_scene_assets(scene)

    if scene == "custom":
        expected_file, expected_source = "scene.xml", "imported"
    elif rt_settings.geometry is not None:
        expected_file, expected_source = "scene.xml", "parameterized"
    else:
        expected_file, expected_source = f"{scene}.xml", "builtin"
    if scene_assets.scene_file != expected_file:
        raise ValueError(
            f"RT scene_assets.scene_file 必须为 {expected_file}，"
            f"实际为 {scene_assets.scene_file}"
        )
    if scene_assets.source != expected_source:
        raise ValueError(
            f"RT scene_assets.source 必须为 {expected_source}，"
            f"实际为 {scene_assets.source}"
        )
    resolved = resolve_scene_assets(
        scene_assets.root,
        scene_assets.scene_file,
        source=scene_assets.source,
    )
    if (
        resolved.file_sha256 != scene_assets.file_sha256
        or resolved.bundle_sha256 != scene_assets.bundle_sha256
    ):
        raise ValueError("RT scene assets changed after resolution")
    if expected_source == "builtin":
        packaged = resolve_builtin_scene_assets(scene)
        if (
            resolved.file_sha256 != packaged.file_sha256
            or resolved.bundle_sha256 != packaged.bundle_sha256
        ):
            raise ValueError("RT built-in scene root does not match packaged scene assets")
    return resolved


def _runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(package)
        for name, package in (
            ("sionna", "sionna"),
            ("sionna_rt", "sionna-rt"),
            ("mitsuba", "mitsuba"),
            ("drjit", "drjit"),
            ("torch", "torch"),
        )
    }


def _json_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cp_energy_diagnostics(
    taps_beam: np.ndarray,
    *,
    l_min: int,
    cyclic_prefix_length: int,
) -> tuple[
    list[list[int | None]],
    list[list[list[int] | None]],
    list[list[float]],
    bool,
]:
    spans: list[list[int | None]] = []
    intervals: list[list[list[int] | None]] = []
    outside_energy_ratios: list[list[float]] = []
    sufficient = True
    cp_window_length = cyclic_prefix_length + 1
    for beam in range(taps_beam.shape[0]):
        beam_spans: list[int | None] = []
        beam_intervals: list[list[int] | None] = []
        beam_outside_ratios: list[float] = []
        for user in range(taps_beam.shape[1]):
            energy = np.abs(taps_beam[beam, user]) ** 2
            total = float(np.sum(energy))
            if total == 0.0:
                beam_spans.append(None)
                beam_intervals.append(None)
                beam_outside_ratios.append(0.0)
                continue
            shortest_span = energy.size
            shortest_interval: list[int] | None = None
            for start in range(energy.size):
                accumulated = 0.0
                for stop in range(start, energy.size):
                    accumulated += float(energy[stop])
                    if accumulated >= 0.99 * total:
                        span = stop - start
                        if span < shortest_span:
                            shortest_span = span
                            shortest_interval = [l_min + start, l_min + stop]
                        break
            if shortest_interval is None:
                raise RuntimeError("CP energy interval did not capture 99% of tap energy")
            if cp_window_length >= energy.size:
                outside_ratio = 0.0
            else:
                prefix = np.concatenate(([0.0], np.cumsum(energy, dtype=np.float64)))
                window_energy = prefix[cp_window_length:] - prefix[:-cp_window_length]
                outside_ratio = max(0.0, 1.0 - float(np.max(window_energy)) / total)
            beam_spans.append(shortest_span)
            beam_intervals.append(shortest_interval)
            beam_outside_ratios.append(outside_ratio)
            sufficient = sufficient and shortest_span <= cyclic_prefix_length
        spans.append(beam_spans)
        intervals.append(beam_intervals)
        outside_energy_ratios.append(beam_outside_ratios)
    return spans, intervals, outside_energy_ratios, sufficient

def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference.ravel()))
    if denom == 0.0:
        return 0.0 if np.linalg.norm(actual.ravel()) == 0.0 else math.inf
    return float(np.linalg.norm((actual - reference).ravel()) / denom)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)
