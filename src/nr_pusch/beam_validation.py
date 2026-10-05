"""Independent geometric and electromagnetic checks for RT beam channels."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from typing import Any, Callable

import numpy as np
import torch
from sionna.phy.ofdm import OFDMDemodulator

from nr_pusch.beam_simulation import (
    _data_subcarrier_indices,
    simulate_rt_beam_bler,
)
from nr_pusch.beamforming import (
    combine_beams,
    compute_beam_metrics,
    make_beam_weights,
    user_power_scales,
)
from nr_pusch.config import TxSettings
from nr_pusch.noise import add_correlated_awgn, beam_noise_covariance
from nr_pusch.rt_channel import (
    NrPuschRtBeamChannel,
    RtBeamSnapshot,
    _arrays_sha256,
    _json_sha256,
    _read_tx_resource_geometry,
    _runtime_versions,
    _snapshot_arrays,
    _validate_snapshot,
    prepare_rt_beam_snapshot,
)
from nr_pusch.rt_config import RtBeamSettings, _toml_value
from nr_pusch.rt_scene_assets import (
    build_parameterized_scene_bundle,
    bundle_sha256_for,
    resolve_builtin_scene_assets,
)
from nr_pusch.simulation_config import BlerSettings
from nr_pusch.transmitter import NrPuschTx


_SPEED_OF_LIGHT_M_S = 299_792_458.0
_VACUUM_PERMITTIVITY_F_M = 8.854_187_8128e-12
_SPECULAR_INTERACTION = 1


def single_layer_slab_reflection_coefficients(
    *,
    frequency_hz: float,
    thickness_m: float,
    relative_permittivity: float,
    conductivity_s_m: float,
    cos_incidence: float,
) -> tuple[complex, complex]:
    """Return the ITU-R P.2040-4 finite-slab TE/TM reflection coefficients."""
    values = (frequency_hz, thickness_m, relative_permittivity, conductivity_s_m, cos_incidence)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("slab Fresnel inputs must be finite")
    if frequency_hz <= 0.0 or thickness_m < 0.0 or relative_permittivity < 1.0:
        raise ValueError("slab Fresnel frequency, thickness, or permittivity is invalid")
    if conductivity_s_m < 0.0 or not 0.0 <= cos_incidence <= 1.0:
        raise ValueError("slab Fresnel conductivity or incidence cosine is invalid")

    wavelength_m = _SPEED_OF_LIGHT_M_S / frequency_hz
    omega = 2.0 * math.pi * frequency_hz
    eta = relative_permittivity - 1j * conductivity_s_m / (
        omega * _VACUUM_PERMITTIVITY_F_M
    )
    sin_squared = 1.0 - cos_incidence * cos_incidence
    transmitted_cosine = np.sqrt(eta - sin_squared + 0.0j)
    r_te_interface = (cos_incidence - transmitted_cosine) / (
        cos_incidence + transmitted_cosine
    )
    r_tm_interface = (eta * cos_incidence - transmitted_cosine) / (
        eta * cos_incidence + transmitted_cosine
    )
    q = 2.0 * math.pi * thickness_m / wavelength_m * transmitted_cosine
    round_trip = np.exp(-2j * q)
    r_te = r_te_interface * (1.0 - round_trip) / (
        1.0 - r_te_interface**2 * round_trip
    )
    r_tm = r_tm_interface * (1.0 - round_trip) / (
        1.0 - r_tm_interface**2 * round_trip
    )
    return complex(r_te), complex(r_tm)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("path geometry contains a zero or non-finite direction")
    return vector / norm


def _implicit_theta_basis(direction: np.ndarray) -> np.ndarray:
    """Sionna's non-polar implicit Jones basis: spherical theta-hat."""
    x, y, z = _unit(direction)
    horizontal = math.hypot(x, y)
    if horizontal <= 1e-7:
        # Match Sionna's deterministic pole fallback (-x projected transverse).
        fallback = np.asarray([-1.0, 0.0, 0.0])
        fallback -= np.dot(fallback, (x, y, z)) * np.asarray([x, y, z])
        return _unit(fallback)
    return np.asarray([z * x / horizontal, z * y / horizontal, -horizontal])


def _basis_rotation(
    direction: np.ndarray,
    current_basis: np.ndarray,
    target_basis: np.ndarray,
) -> np.ndarray:
    """Rotate Jones components using Sionna's implicit-basis convention."""
    k = _unit(direction)
    current = _unit(current_basis)
    target = _unit(target_basis)
    cosine = float(np.dot(current, target))
    sine = float(np.dot(k, np.cross(current, target)))
    return np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.complex128)


def _v_iso_reflection_coefficient(
    incident_direction: np.ndarray,
    reflected_direction: np.ndarray,
    surface_normal: np.ndarray,
    reflection_te: complex,
    reflection_tm: complex,
) -> complex:
    """Apply the slab Jones matrix between Sionna V-isotropic antenna bases."""
    k_in = _unit(incident_direction)
    k_out = _unit(reflected_direction)
    normal = _unit(surface_normal)
    te_basis = _unit(np.cross(k_in, normal))
    input_rotation = _basis_rotation(
        k_in, _implicit_theta_basis(k_in), te_basis
    )
    output_rotation = _basis_rotation(
        k_out, te_basis, _implicit_theta_basis(k_out)
    )
    jones = output_rotation @ np.diag([reflection_te, reflection_tm]) @ input_rotation
    return complex(jones[0, 0])


def _path_interactions_and_vertices(
    snapshot: RtBeamSnapshot,
    user_index: int,
    path_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    interactions = np.asarray(snapshot.path_interactions)
    vertices = np.asarray(snapshot.path_vertices_m)
    axes = snapshot.metadata["axis_order"]["path_interactions"]
    if axes == ["depth", "receiver", "user", "path"]:
        return interactions[:, 0, user_index, path_index], vertices[:, 0, user_index, path_index]
    if axes == ["depth", "receiver", "element", "user", "tx_antenna", "path"]:
        return interactions[:, 0, 0, user_index, 0, path_index], vertices[:, 0, 0, user_index, 0, path_index]
    raise ValueError(f"unsupported RT interaction axes: {axes}")


def _surface_for_vertex(vertex: np.ndarray) -> tuple[str, np.ndarray] | None:
    if abs(float(vertex[2])) <= 1e-3:
        return "ground-concrete", np.asarray([0.0, 0.0, 1.0])
    if abs(float(vertex[1]) - 150.0) <= 1e-3:
        return "wall-brick", np.asarray([0.0, -1.0, 0.0])
    return None


def _image_source(position: np.ndarray, surface: str) -> np.ndarray:
    image = np.asarray(position, dtype=np.float64).copy()
    if surface == "ground-concrete":
        image[2] = -image[2]
    elif surface == "wall-brick":
        image[1] = 300.0 - image[1]
    else:
        raise ValueError(f"unknown RT reflector {surface!r}")
    return image


def validate_rt_multipath_snapshot(
    tx_settings: TxSettings,
    rt_settings: RtBeamSettings,
    snapshot: RtBeamSnapshot,
) -> dict[str, Any]:
    """Check reflected path geometry and complex V-polarized slab response."""
    rt_settings.validate_transmitter(tx_settings)
    if rt_settings.rt.scene != "ground_wall":
        raise ValueError("multipath reflection oracle requires the ground_wall scene")
    if not rt_settings.rt.synthetic_array:
        raise ValueError("multipath slab oracle currently requires synthetic_array=true")

    carrier_hz = rt_settings.rt.carrier_frequency_hz
    wavelength_m = _SPEED_OF_LIGHT_M_S / carrier_hz
    receiver = np.asarray(rt_settings.receiver.position_m, dtype=np.float64)
    materials = snapshot.metadata.get("scene_materials", {})
    required_materials = {"ground-concrete", "wall-brick"}
    if not required_materials.issubset(materials):
        raise ValueError(f"RT snapshot is missing material parameters: {sorted(required_materials - set(materials))}")

    checks: list[dict[str, Any]] = []
    for user_index, user in enumerate(rt_settings.users):
        source = np.asarray(user.position_m, dtype=np.float64)
        valid_paths = np.flatnonzero(np.any(snapshot.path_valid[:, user_index, :], axis=0))
        direct_paths: list[int] = []
        reflected_paths: list[tuple[int, str, np.ndarray]] = []
        for path_index in valid_paths:
            interactions, vertices = _path_interactions_and_vertices(
                snapshot, user_index, int(path_index)
            )
            active = interactions != 0
            active_types = interactions[active]
            active_vertices = vertices[active]
            if not active_types.size:
                direct_paths.append(int(path_index))
                continue
            if not np.all(active_types == _SPECULAR_INTERACTION):
                continue
            if active_types.size != 1:
                continue
            surface = _surface_for_vertex(active_vertices[0])
            if surface is not None:
                reflected_paths.append((int(path_index), surface[0], active_vertices[0]))

        if len(direct_paths) != 1:
            raise ValueError(f"{user.name}: expected one direct path, got {direct_paths}")
        if not reflected_paths:
            raise ValueError(f"{user.name}: no single-specular ground/wall path found")

        direct_index = direct_paths[0]
        direct_distance = float(np.linalg.norm(source - receiver))
        direct_observed_m = float(snapshot.path_tau_s[0, user_index, direct_index] * _SPEED_OF_LIGHT_M_S)
        direct_delay_relative_error = abs(direct_observed_m / direct_distance - 1.0)
        if direct_delay_relative_error > 1e-5:
            raise ValueError(
                f"{user.name}: direct-path geometry error {direct_delay_relative_error:.6g} exceeds 1e-5"
            )

        element_position = np.asarray(snapshot.element_positions_m[0], dtype=np.float64)
        direct_arrival = _unit(source - receiver)
        direct_center = snapshot.path_a[0, user_index, direct_index] * np.exp(
            -2j * math.pi * np.dot(element_position, direct_arrival) / wavelength_m
        )
        if abs(direct_center) == 0.0:
            raise ValueError(f"{user.name}: direct path coefficient is zero")

        for path_index, material_name, vertex in reflected_paths:
            image = _image_source(source, material_name)
            expected_distance = float(np.linalg.norm(image - receiver))
            observed_distance = float(snapshot.path_tau_s[0, user_index, path_index] * _SPEED_OF_LIGHT_M_S)
            delay_relative_error = abs(observed_distance / expected_distance - 1.0)

            incident = _unit(vertex - source)
            outgoing = _unit(receiver - vertex)
            surface = _surface_for_vertex(vertex)
            if surface is None:
                raise ValueError(f"{user.name}: path {path_index} vertex does not match a reflector")
            cos_incidence = float(np.clip(-np.dot(incident, surface[1]), 0.0, 1.0))
            material = materials[material_name]
            r_te, r_tm = single_layer_slab_reflection_coefficients(
                frequency_hz=carrier_hz,
                thickness_m=float(material["thickness_m"]),
                relative_permittivity=float(material["relative_permittivity"]),
                conductivity_s_m=float(material["conductivity_s_per_m"]),
                cos_incidence=cos_incidence,
            )
            gamma_expected = _v_iso_reflection_coefficient(
                incident,
                outgoing,
                surface[1],
                r_te,
                r_tm,
            )
            arrival = _unit(vertex - receiver)
            reflected_center = snapshot.path_a[0, user_index, path_index] * np.exp(
                -2j * math.pi * np.dot(element_position, arrival) / wavelength_m
            )
            gamma_observed = (
                reflected_center
                / direct_center
                * (expected_distance / direct_distance)
                * np.exp(2j * math.pi * carrier_hz * (expected_distance - direct_distance) / _SPEED_OF_LIGHT_M_S)
            )
            if abs(gamma_expected) == 0.0:
                amplitude_relative_error = math.inf
                phase_error_rad = math.inf
            else:
                ratio = gamma_observed / gamma_expected
                amplitude_relative_error = abs(abs(ratio) - 1.0)
                phase_error_rad = abs(float(np.angle(ratio)))
            passed = (
                delay_relative_error <= 1e-5
                and amplitude_relative_error <= 0.02
                and phase_error_rad <= 0.03
            )
            checks.append(
                {
                    "check": "single_specular_reflection",
                    "user": user.name,
                    "path_index": path_index,
                    "material": material_name,
                    "formula": "finite-slab P.2040-4 TE/TM reflection, world-implicit Jones projection, image-source path length",
                    "expected_distance_m": expected_distance,
                    "observed_distance_m": observed_distance,
                    "delay_relative_error": delay_relative_error,
                    "incidence_cosine": cos_incidence,
                    "fresnel_te": [r_te.real, r_te.imag],
                    "fresnel_tm": [r_tm.real, r_tm.imag],
                    "expected_v_pol_coefficient": [gamma_expected.real, gamma_expected.imag],
                    "observed_v_pol_coefficient": [gamma_observed.real, gamma_observed.imag],
                    "amplitude_relative_error": amplitude_relative_error,
                    "phase_error_rad": phase_error_rad,
                    "thresholds": {
                        "delay_relative_error": 1e-5,
                        "amplitude_relative_error": 0.02,
                        "phase_error_rad": 0.03,
                    },
                    "passed": passed,
                }
            )

    if not checks:
        raise ValueError("ground_wall scene produced no single-reflection checks")
    return {
        "stage": "multipath",
        "scene": rt_settings.rt.scene,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "path_count_per_user": snapshot.metadata["valid_path_count_per_user"],
        "scene_materials": materials,
    }


def _path_geometry_key(
    snapshot: RtBeamSnapshot,
    user_index: int,
    path_index: int,
) -> tuple[tuple[int, int, int, int], ...]:
    interactions, vertices = _path_interactions_and_vertices(
        snapshot, user_index, path_index
    )
    key = [
        (
            int(interaction),
            *np.rint(vertex / 1e-4).astype(np.int64).tolist(),
        )
        for interaction, vertex in zip(interactions, vertices)
        if int(interaction) != 0
    ]
    return tuple(sorted(key))


def _path_records_by_geometry(
    snapshot: RtBeamSnapshot,
    user_index: int,
) -> dict[tuple[tuple[int, int, int, int], ...], int]:
    valid_paths = np.flatnonzero(
        np.any(snapshot.path_valid[:, user_index, :], axis=0)
    )
    records = {
        _path_geometry_key(snapshot, user_index, int(path_index)): int(path_index)
        for path_index in valid_paths
    }
    if len(records) != len(valid_paths):
        raise ValueError(
            f"{snapshot.metadata['users'][user_index]}: duplicate path geometries"
        )
    return records


def validate_rt_path_convergence(
    lower_budget_snapshot: RtBeamSnapshot,
    higher_budget_snapshot: RtBeamSnapshot,
) -> dict[str, Any]:
    """Compare geometry-matched paths and CFR after doubling ray samples."""
    lower_solver = lower_budget_snapshot.metadata["solver"]
    higher_solver = higher_budget_snapshot.metadata["solver"]
    lower_samples = int(lower_solver["samples_per_src"])
    higher_samples = int(higher_solver["samples_per_src"])
    if lower_samples <= 0 or higher_samples != 2 * lower_samples:
        raise ValueError(
            "path convergence requires positive samples_per_src then exactly double, "
            f"got {lower_samples} then {higher_samples}"
        )
    if (
        lower_solver["seed"] != higher_solver["seed"]
        or lower_solver["deterministic"] is not True
        or higher_solver["deterministic"] is not True
        or lower_solver["max_depth"] != higher_solver["max_depth"]
        or lower_solver["max_num_paths_per_src"] != higher_solver["max_num_paths_per_src"]
        or lower_solver["synthetic_array"] != higher_solver["synthetic_array"]
    ):
        raise ValueError(
            "path convergence requires identical deterministic solver settings except samples"
        )
    if (
        lower_budget_snapshot.metadata["scene_bundle_sha256"]
        != higher_budget_snapshot.metadata["scene_bundle_sha256"]
    ):
        raise ValueError("path convergence snapshots have different scene asset bundles")
    if lower_budget_snapshot.metadata["users"] != higher_budget_snapshot.metadata["users"]:
        raise ValueError("path convergence snapshots have different UE order")

    checks: list[dict[str, Any]] = []
    for user_index, user in enumerate(lower_budget_snapshot.metadata["users"]):
        lower_records = _path_records_by_geometry(lower_budget_snapshot, user_index)
        higher_records = _path_records_by_geometry(higher_budget_snapshot, user_index)
        if set(lower_records) != set(higher_records):
            checks.append(
                {
                    "check": "path_geometry_set_stable",
                    "user": user,
                    "lower_path_count": len(lower_records),
                    "higher_path_count": len(higher_records),
                    "passed": False,
                }
            )
            continue
        for geometry_key, lower_path in lower_records.items():
            higher_path = higher_records[geometry_key]
            lower_delay = float(
                lower_budget_snapshot.path_tau_s[0, user_index, lower_path]
            )
            higher_delay = float(
                higher_budget_snapshot.path_tau_s[0, user_index, higher_path]
            )
            delay_relative_error = abs(higher_delay / lower_delay - 1.0)
            lower_coefficient = lower_budget_snapshot.path_a[
                0, user_index, lower_path
            ]
            higher_coefficient = higher_budget_snapshot.path_a[
                0, user_index, higher_path
            ]
            coefficient_relative_error = (
                abs(higher_coefficient - lower_coefficient) / abs(lower_coefficient)
                if abs(lower_coefficient) > 0.0
                else math.inf
            )
            checks.append(
                {
                    "check": "path_coefficient_convergence",
                    "user": user,
                    "geometry": geometry_key,
                    "lower_path_index": lower_path,
                    "higher_path_index": higher_path,
                    "delay_relative_error": delay_relative_error,
                    "coefficient_relative_error": coefficient_relative_error,
                    "passed": delay_relative_error <= 1e-5
                    and coefficient_relative_error <= 0.01,
                }
            )

    lower_cfr = lower_budget_snapshot.h_beam
    higher_cfr = higher_budget_snapshot.h_beam
    if lower_cfr.shape != higher_cfr.shape:
        cfr_relative_error = math.inf
    else:
        denominator = float(np.linalg.norm(lower_cfr))
        cfr_relative_error = (
            float(np.linalg.norm(higher_cfr - lower_cfr) / denominator)
            if denominator > 0.0
            else math.inf
        )
    checks.append(
        {
            "check": "summed_cfr_convergence",
            "relative_error": cfr_relative_error,
            "threshold": 0.01,
            "passed": cfr_relative_error <= 0.01,
        }
    )
    return {
        "stage": "path_convergence",
        "samples_per_src": [lower_samples, higher_samples],
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def rt_tap_frequency_response(snapshot: RtBeamSnapshot) -> np.ndarray:
    """Evaluate stored beam-domain taps at the snapshot's baseband frequencies."""
    l_min = int(snapshot.metadata["l_min"])
    l_max = int(snapshot.metadata["l_max"])
    sample_rate_hz = int(snapshot.metadata["sample_rate_hz"])
    tap_indices = np.arange(l_min, l_max + 1, dtype=np.float64)
    if snapshot.taps_beam.shape[-1] != tap_indices.size:
        raise ValueError("RT snapshot tap count does not match its l_min/l_max")
    if sample_rate_hz <= 0:
        raise ValueError("RT snapshot sample rate must be positive")
    phase = np.exp(
        -2j
        * np.pi
        * snapshot.frequencies_hz[None, None, :, None]
        * tap_indices[None, None, None, :]
        / sample_rate_hz
    )
    return np.sum(snapshot.taps_beam[:, :, None, :] * phase, axis=-1)


def validate_rt_time_domain(
    tx_settings: TxSettings,
    snapshot: RtBeamSnapshot,
    *,
    device: str | None = None,
    seed: int = 13,
) -> dict[str, Any]:
    """Check finite-tap CFR, CP support, and native OFDM time/frequency equivalence."""
    tx_settings.validate()
    expected_users = [user.name for user in tx_settings.users]
    if snapshot.metadata.get("users") != expected_users:
        raise ValueError("RT snapshot UE order does not match TX settings")
    tap_cfr = rt_tap_frequency_response(snapshot)
    if tap_cfr.shape != snapshot.h_beam.shape:
        raise ValueError("RT tap CFR shape does not match the direct path CFR")

    tap_window_ratio = float(
        snapshot.metadata["tap_out_of_window_path_energy_ratio"]
    )
    direct_cfr_relative_error = _relative_error(tap_cfr, snapshot.h_beam)
    cp_outside_ratio = float(snapshot.metadata["cp_outside_energy_ratio_max"])
    cp_sufficient = bool(snapshot.metadata["cp_sufficient"])
    checks = [
        {
            "check": "tap_window_outside_energy",
            "relative_energy": tap_window_ratio,
            "threshold": 0.01,
            "passed": tap_window_ratio <= 0.01,
        },
        {
            "check": "direct_rt_cfr_vs_finite_tap_cfr",
            "relative_error": direct_cfr_relative_error,
            "threshold": 0.01,
            "passed": direct_cfr_relative_error <= 0.01,
        },
        {
            "check": "cp_outside_effective_energy",
            "relative_energy": cp_outside_ratio,
            "threshold": 0.01,
            "fd_equivalence_eligible": cp_sufficient,
            "passed": cp_outside_ratio <= 0.01 if cp_sufficient else None,
        },
    ]

    transmitter = NrPuschTx(tx_settings, device=device)
    resource_grid = transmitter._tx_freq.resource_grid
    generated = transmitter.generate(batch_size=1, seed=seed)
    channel = NrPuschRtBeamChannel(snapshot, device=transmitter.device)
    time_result = channel.apply(generated.iq, generated.sample_rate_hz)
    fd_td_relative_rms: float | None = None
    fd_td_status = "skipped_cp_insufficient"
    if cp_sufficient:
        demodulator = OFDMDemodulator(
            fft_size=int(resource_grid.fft_size),
            l_min=int(snapshot.metadata["l_min"]),
            cyclic_prefix_length=resource_grid.cyclic_prefix_length,
            device=transmitter.device,
        )
        demodulated = demodulator(time_result.iq)
        channel_tensor = torch.as_tensor(
            tap_cfr,
            dtype=generated.frequency_grid.dtype,
            device=transmitter.device,
        ).permute(1, 0, 2)
        powers = torch.as_tensor(
            np.sqrt(user_power_scales(snapshot.power_scale_db)),
            dtype=generated.frequency_grid.real.dtype,
            device=transmitter.device,
        )
        user_grid = generated.frequency_grid[:, :, 0]
        expected_frequency_grid = (
            user_grid[:, :, None, :, :]
            * channel_tensor[None, :, :, None, :]
            * powers[None, :, None, None, None]
        ).sum(dim=1)
        if demodulated.shape != expected_frequency_grid.shape:
            raise ValueError(
                "RT TD-demodulated grid shape does not match the finite-tap frequency grid: "
                f"{tuple(demodulated.shape)} != {tuple(expected_frequency_grid.shape)}"
            )
        fd_td_relative_rms = _relative_error(
            demodulated.detach().cpu().numpy(),
            expected_frequency_grid.detach().cpu().numpy(),
        )
        fd_td_status = "passed" if fd_td_relative_rms <= 1e-5 else "failed"
        checks.append(
            {
                "check": "native_ofdm_fd_td_equivalence",
                "relative_rms_error": fd_td_relative_rms,
                "threshold": 1e-5,
                "passed": fd_td_relative_rms <= 1e-5,
            }
        )

    required_checks_passed = all(
        check["passed"] is not False
        for check in checks
        if check["check"] != "cp_outside_effective_energy"
    )
    return {
        "stage": "time",
        "tap_frequency_response_axes": ["beam", "user", "fft_bin"],
        "tap_window_outside_energy_ratio": tap_window_ratio,
        "cp_energy_interval_samples_by_beam_user": snapshot.metadata[
            "cp_energy_interval_samples_by_beam_user"
        ],
        "cp_energy_interval_bounds_samples_by_beam_user": snapshot.metadata.get(
            "cp_energy_interval_bounds_samples_by_beam_user"
        ),
        "cp_outside_energy_ratio_by_beam_user": snapshot.metadata[
            "cp_outside_energy_ratio_by_beam_user"
        ],
        "cp_sufficient": cp_sufficient,
        "cyclic_prefix_length_samples": int(
            snapshot.metadata["cyclic_prefix_length_samples"]
        ),
        "direct_rt_cfr_truncation_relative_error": direct_cfr_relative_error,
        "fd_td_relative_rms_error": fd_td_relative_rms,
        "fd_td_status": fd_td_status,
        "checks": checks,
        "passed": required_checks_passed
        and (fd_td_relative_rms is None or fd_td_relative_rms <= 1e-5),
    }


def validate_rt_web_snapshot(
    tx_settings: TxSettings,
    rt_settings: RtBeamSettings,
    snapshot: RtBeamSnapshot,
    *,
    higher_budget_snapshot: RtBeamSnapshot | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Gate the bounded Web frequency-domain approximation without changing strict reports."""
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add_check(
        name: str,
        passed: bool | None,
        *,
        actual: Any = None,
        threshold: Any = None,
        reason: str | None = None,
        required: bool = True,
    ) -> None:
        check: dict[str, Any] = {"check": name, "passed": passed, "required": required}
        if actual is not None:
            check["actual"] = actual
        if threshold is not None:
            check["threshold"] = threshold
        if reason is not None:
            check["reason"] = reason
        checks.append(check)

    strict_time_report: dict[str, Any] | None = None
    convergence: dict[str, Any] | None = None
    reflection_oracle: dict[str, Any]
    los_oracle: dict[str, Any]
    snapshot_hash: str | None = None
    q_ref: float | None = None

    def identity_error(
        candidate: RtBeamSnapshot,
        expected_geometry: dict[str, Any],
        resource_grid: Any,
        *,
        allow_doubled_budget: bool = False,
    ) -> str | None:
        metadata = candidate.metadata
        try:
            if metadata.get("format_version") != 2:
                raise ValueError("RT snapshot format_version 不匹配；请重新 prepare")
            if metadata.get("versions") != _runtime_versions():
                raise ValueError("RT snapshot 软件版本与当前运行环境不匹配")
            stored_rt = metadata.get("rt_config")
            if not isinstance(stored_rt, dict):
                raise ValueError("RT snapshot 缺少有效 RT 配置")
            expected_rt = rt_settings.to_dict()
            if allow_doubled_budget:
                expected_samples = int(stored_rt.get("rt", {}).get("samples_per_src", 0))
                expected_rt["rt"]["samples_per_src"] = expected_samples
                if expected_samples != 2 * rt_settings.rt.samples_per_src:
                    raise ValueError("higher-budget snapshot samples_per_src 必须恰好翻倍")
            if _json_sha256(stored_rt) != _json_sha256(expected_rt):
                raise ValueError("RT snapshot 配置与当前 RT settings 不匹配")
            if metadata.get("users") != [user.name for user in tx_settings.users]:
                raise ValueError("RT snapshot UE 顺序与 TX 配置不匹配")
            if metadata.get("tx_resource_geometry") != expected_geometry:
                raise ValueError("RT snapshot TX resource-grid/DMRS geometry 与当前 TX 配置不匹配")
            if metadata.get("tx_resource_geometry_sha256") != _json_sha256(expected_geometry):
                raise ValueError("RT snapshot TX resource geometry 摘要损坏")
            scene = rt_settings.rt.scene
            if metadata.get("scene") != scene:
                raise ValueError("RT snapshot scene 与当前 RT settings 不匹配")
            if scene == "custom":
                expected_file, expected_source = "scene.xml", "imported"
            elif rt_settings.geometry is not None:
                expected_file, expected_source = "scene.xml", "parameterized"
            else:
                expected_file, expected_source = f"{scene}.xml", "builtin"
            if metadata.get("scene_file") != expected_file or metadata.get("scene_source") != expected_source:
                raise ValueError("RT snapshot scene file/source 与当前 RT settings 不匹配")
            scene_hashes = metadata.get("scene_asset_sha256")
            bundle_hash = metadata.get("scene_bundle_sha256")
            if not isinstance(scene_hashes, dict) or expected_file not in scene_hashes:
                raise ValueError("RT snapshot scene asset hashes 缺失或不完整")
            if bundle_sha256_for(scene_hashes) != bundle_hash:
                raise ValueError("RT snapshot scene bundle hash 与文件摘要不一致")
            if expected_source == "builtin":
                packaged = resolve_builtin_scene_assets(scene)
                if scene_hashes != packaged.file_sha256 or bundle_hash != packaged.bundle_sha256:
                    raise ValueError("RT snapshot 场景资源 hash 与当前 package assets 不匹配")
            elif expected_source == "parameterized":
                expected_bundle = build_parameterized_scene_bundle(scene, rt_settings.geometry)
                if scene_hashes != expected_bundle.file_sha256 or bundle_hash != expected_bundle.bundle_sha256:
                    raise ValueError("RT snapshot parameterized scene hash 与当前 geometry 不匹配")
            expected_config_hash = _json_sha256(
                {
                    "rt_config": stored_rt,
                    "tx_resource_geometry": expected_geometry,
                    "scene_bundle_sha256": bundle_hash,
                }
            )
            if metadata.get("config_sha256") != expected_config_hash:
                raise ValueError("RT snapshot 配置摘要不匹配")
            if metadata.get("array_sha256") is not None and metadata["array_sha256"] != _arrays_sha256(
                _snapshot_arrays(candidate)
            ):
                raise ValueError("RT snapshot NPZ 数组摘要校验失败")
            _validate_snapshot(
                candidate,
                tx_settings,
                expected_geometry,
                resource_grid=resource_grid,
            )
        except Exception as exc:
            return str(exc)
        return None

    try:
        tx_settings.validate()
        rt_settings.validate()
        rt_settings.validate_transmitter(tx_settings)
        expected_geometry, resource_grid = _read_tx_resource_geometry(tx_settings)
        lower_identity_error = identity_error(snapshot, expected_geometry, resource_grid)
    except Exception as exc:
        lower_identity_error = str(exc)
        expected_geometry = {}
        resource_grid = None
    identity_ok = lower_identity_error is None
    add_check(
        "snapshot_configuration_software_array_asset_identity",
        identity_ok,
        actual=(
            {
                "format_version": snapshot.metadata.get("format_version"),
                "scene_bundle_sha256": snapshot.metadata.get("scene_bundle_sha256"),
            }
            if identity_ok else None
        ),
        reason=lower_identity_error,
    )

    if identity_ok:
        try:
            finite_arrays = all(
                np.all(np.isfinite(np.asarray(getattr(snapshot, key))))
                for key in (
                    "path_a", "path_tau_s", "ta_s", "element_positions_m", "weights",
                    "h_ant", "h_beam", "taps_ant", "taps_beam", "frequencies_hz",
                    "path_theta_r_rad", "path_phi_r_rad", "path_vertices_m",
                    "path_interactions",
                )
            )
        except (TypeError, ValueError):
            finite_arrays = False
        add_check("all_snapshot_numeric_arrays_finite", finite_arrays)
        add_check(
            "all_snapshot_numeric_metadata_finite",
            _web_values_finite(snapshot.metadata),
        )
        for user_index, user in enumerate(rt_settings.users):
            path_count = int(np.any(snapshot.path_valid[:, user_index, :], axis=0).sum())
            add_check(
                f"valid_path_for_{user.name}",
                path_count >= 1,
                actual=path_count,
                threshold=1,
            )
        try:
            tx = NrPuschTx(tx_settings, device="cpu")
            data_bins = _data_subcarrier_indices(tx).cpu().numpy()
            powers = np.asarray(user_power_scales(snapshot.power_scale_db), dtype=np.float64)
            q_ref = float(
                np.mean(
                    np.abs(snapshot.h_beam[0, 0, data_bins] * math.sqrt(powers[0])) ** 2
                )
            )
            q_ref_ok = math.isfinite(q_ref) and q_ref > 0.0
        except Exception as exc:
            q_ref_ok = False
            q_ref = None
            q_ref_error = str(exc)
        else:
            q_ref_error = None if q_ref_ok else "Q_ref must be positive and finite"
        add_check(
            "positive_finite_reference_channel_power_q_ref",
            q_ref_ok,
            actual=q_ref,
            threshold="> 0",
            reason=q_ref_error,
        )

        try:
            beam_report = validate_rt_beam_structure(snapshot)
            required_beam_checks = [
                item for item in beam_report["checks"]
                if item["check"] != "per_beam_steering_main_lobe"
            ]
            beam_ok = all(item["passed"] is True for item in required_beam_checks)
            add_check(
                "beam_matrix_composition_gram_and_weight_reconstruction",
                beam_ok,
                actual=required_beam_checks,
                reason=None if beam_ok else "required beam structure check failed",
            )
            steering = next(
                item for item in beam_report["checks"]
                if item["check"] == "per_beam_steering_main_lobe"
            )
            add_check(
                "per_beam_steering_main_lobe_diagnostic",
                None,
                actual=steering,
                reason="diagnostic only; quantization/pointing does not block Web BLER",
                required=False,
            )
        except Exception as exc:
            add_check(
                "beam_matrix_composition_gram_and_weight_reconstruction",
                False,
                reason=str(exc),
            )

        try:
            weights = torch.as_tensor(snapshot.weights, dtype=torch.complex128, device="cpu")
            covariance = beam_noise_covariance(
                weights,
                array_variance=1.0,
                post_variance=rt_settings.noise.post_combiner_ratio,
            ).numpy()
            eigenvalues = np.linalg.eigvalsh(covariance)
            covariance_ok = bool(
                np.all(np.isfinite(covariance))
                and np.all(np.isfinite(eigenvalues))
                and float(eigenvalues.min()) >= -1e-12
            )
            covariance_actual: Any = {
                "minimum_eigenvalue": float(eigenvalues.min()),
                "post_combiner_ratio": rt_settings.noise.post_combiner_ratio,
                "matrix_real": covariance.real.tolist(),
                "matrix_imag": covariance.imag.tolist(),
            }
            covariance_error = None
        except Exception as exc:
            covariance_ok = False
            covariance_actual = None
            covariance_error = str(exc)
        add_check(
            "physical_noise_covariance_positive_semidefinite",
            covariance_ok,
            actual=covariance_actual,
            threshold="minimum eigenvalue >= -1e-12",
            reason=covariance_error,
        )

        try:
            strict_time_report = validate_rt_time_domain(
                tx_settings, snapshot, device="cpu"
            )
        except Exception as exc:
            strict_time_report = {
                "stage": "time",
                "passed": False,
                "error": str(exc),
            }
        cp_sufficient = strict_time_report.get("cp_sufficient") is True
        add_check(
            "cyclic_prefix_sufficient",
            cp_sufficient,
            actual=strict_time_report.get("cp_sufficient"),
            threshold=True,
        )
        tap_window = strict_time_report.get("tap_window_outside_energy_ratio")
        tap_window_ok = (
            isinstance(tap_window, (int, float))
            and math.isfinite(float(tap_window))
            and 0.0 <= float(tap_window) <= 0.01
        )
        add_check(
            "tap_window_outside_energy",
            tap_window_ok,
            actual=tap_window,
            threshold=0.01,
            reason=None if tap_window_ok else "missing, non-finite, negative, or over threshold",
        )
        cp_outside = snapshot.metadata.get("cp_outside_energy_ratio_max")
        cp_outside_ok = (
            isinstance(cp_outside, (int, float))
            and math.isfinite(float(cp_outside))
            and 0.0 <= float(cp_outside) <= 0.01
        )
        add_check(
            "cyclic_prefix_outside_effective_energy",
            cp_outside_ok,
            actual=cp_outside,
            threshold=0.01,
            reason=None if cp_outside_ok else "missing, non-finite, negative, or over threshold",
        )
        cfr_error = strict_time_report.get("direct_rt_cfr_truncation_relative_error")
        cfr_ok = (
            isinstance(cfr_error, (int, float))
            and math.isfinite(float(cfr_error))
            and 0.0 <= float(cfr_error) <= 0.01
        )
        add_check(
            "direct_cfr_vs_finite_tap_cfr",
            cfr_ok,
            actual=cfr_error,
            threshold=0.01,
            reason=None if cfr_ok else "missing, non-finite, negative, or over threshold",
        )
        rms = strict_time_report.get("fd_td_relative_rms_error")
        rms_ok = (
            isinstance(rms, (int, float))
            and math.isfinite(float(rms))
            and 0.0 <= float(rms) <= 0.001
        )
        add_check(
            "native_ofdm_fd_td_web_approximation",
            rms_ok,
            actual=rms,
            threshold=0.001,
            reason=None if rms_ok else "missing, non-finite, negative, or over threshold",
        )
        strict_fd_td_passed = (
            isinstance(rms, (int, float))
            and math.isfinite(float(rms))
            and float(rms) <= 1e-5
        )
        if not strict_fd_td_passed and rms_ok:
            warnings.append(
                "严格 FD/TD 1e-5 门槛未通过；网页 BLER 仅在 web-frequency-v1 有界频域近似内成立"
            )
    else:
        strict_fd_td_passed = False
        strict_time_report = {
            "stage": "time",
            "passed": False,
            "skipped_reason": "snapshot identity/array validation failed",
        }
        for name in (
            "all_snapshot_numeric_arrays_finite",
            "all_snapshot_numeric_metadata_finite",
            "valid_path_for_each_user",
            "positive_finite_reference_channel_power_q_ref",
            "beam_matrix_composition_gram_and_weight_reconstruction",
            "physical_noise_covariance_positive_semidefinite",
            "cyclic_prefix_sufficient",
            "tap_window_outside_energy",
            "cyclic_prefix_outside_effective_energy",
            "direct_cfr_vs_finite_tap_cfr",
            "native_ofdm_fd_td_web_approximation",
        ):
            add_check(name, False, reason="snapshot identity/array validation failed")

    asset_names = snapshot.metadata.get("scene_asset_sha256")
    has_mesh = isinstance(asset_names, dict) and any(
        str(path).lower().endswith(".ply") for path in asset_names
    )
    if has_mesh:
        if higher_budget_snapshot is None:
            convergence = {
                "passed": False,
                "error": "mesh scenes require a same-seed trace with doubled samples_per_src",
            }
            add_check(
                "mesh_path_and_cfr_convergence",
                False,
                threshold={"delay_relative_error": 1e-5, "coefficient_relative_error": 0.01, "cfr_relative_error": 0.01},
                reason=convergence["error"],
            )
        else:
            higher_error = identity_error(
                higher_budget_snapshot,
                expected_geometry,
                resource_grid,
                allow_doubled_budget=True,
            ) if identity_ok else "lower-budget snapshot identity failed"
            if higher_error is not None:
                convergence = {"passed": False, "error": higher_error}
            else:
                try:
                    convergence = validate_rt_path_convergence(snapshot, higher_budget_snapshot)
                except Exception as exc:
                    convergence = {"passed": False, "error": str(exc)}
            add_check(
                "mesh_path_and_cfr_convergence",
                convergence.get("passed") is True,
                actual={
                    "passed": convergence.get("passed"),
                    "samples_per_src": convergence.get("samples_per_src"),
                    "check_count": len(convergence.get("checks", [])),
                },
                threshold={"delay_relative_error": 1e-5, "coefficient_relative_error": 0.01, "cfr_relative_error": 0.01},
                reason=convergence.get("error"),
            )
    else:
        add_check(
            "mesh_path_and_cfr_convergence",
            True,
            actual="not required; scene bundle has no PLY mesh",
        )

    if _web_ground_wall_oracle_applies(rt_settings, snapshot):
        try:
            reflection_oracle = validate_rt_multipath_snapshot(
                tx_settings, rt_settings, snapshot
            )
            add_check(
                "built_in_ground_wall_slab_oracle",
                reflection_oracle.get("passed") is True,
                actual={
                    "passed": reflection_oracle.get("passed"),
                    "path_count_per_user": reflection_oracle.get("path_count_per_user"),
                    "check_count": len(reflection_oracle.get("checks", [])),
                },
            )
        except Exception as exc:
            reflection_oracle = {"applicable": True, "passed": False, "error": str(exc)}
            add_check("built_in_ground_wall_slab_oracle", False, reason=str(exc))
    else:
        reflection_oracle = {
            "applicable": False,
            "status": "not_applicable",
            "reason": "oracle only covers the unedited built-in ground_wall geometry",
        }

    if (
        rt_settings.rt.scene == "empty"
        and snapshot.metadata.get("scene_source") == "builtin"
        and rt_settings.geometry is None
        and _web_zero_orientations(rt_settings)
    ):
        try:
            los_oracle = validate_rt_los_snapshot(tx_settings, rt_settings, snapshot)
            add_check(
                "empty_scene_los_friis_oracle",
                los_oracle.get("passed") is True,
                actual={
                    "passed": los_oracle.get("passed"),
                    "check_count": len(los_oracle.get("checks", [])),
                },
            )
        except Exception as exc:
            los_oracle = {"applicable": True, "passed": False, "error": str(exc)}
            add_check("empty_scene_los_friis_oracle", False, reason=str(exc))
    else:
        los_oracle = {
            "applicable": False,
            "status": "not_applicable",
            "reason": (
                "empty-scene Friis oracle requires builtin empty assets and zero BS/UE orientations"
            ),
        }

    try:
        snapshot_hash = snapshot.metadata.get("array_sha256") or _arrays_sha256(
            _snapshot_arrays(snapshot)
        )
    except Exception:
        snapshot_hash = None
    passed = all(
        check["passed"] is True
        for check in checks
        if check.get("required", True)
    )
    report = {
        "format_version": 1,
        "policy": "web-frequency-v1",
        "passed": passed,
        "strict_fd_td_passed": strict_fd_td_passed,
        "checks": checks,
        "warnings": warnings,
        "strict_time_report": strict_time_report,
        "convergence": convergence,
        "reflection_oracle": reflection_oracle,
        "los_oracle": los_oracle,
        "snapshot_hash": snapshot_hash,
        "scene_bundle_sha256": snapshot.metadata.get("scene_bundle_sha256"),
        "reference_channel_power_q_ref": q_ref,
        "device": "cpu",
        "requested_device": device,
    }
    return _web_json_safe(report)


def _web_ground_wall_oracle_applies(
    rt_settings: RtBeamSettings,
    snapshot: RtBeamSnapshot,
) -> bool:
    if (
        rt_settings.rt.scene != "ground_wall"
        or rt_settings.geometry is not None
        or rt_settings.rt.synthetic_array is not True
        or rt_settings.rt.max_depth != 2
        or snapshot.metadata.get("scene_source") != "builtin"
        or rt_settings.receiver.position_m != (0.0, 0.0, 25.0)
        or not _web_zero_orientations(rt_settings)
    ):
        return False
    expected_users = (
        (153.2088886, -128.5575220, 1.5),
        (193.1851652, -51.7638090, 1.5),
        (193.1851652, 51.7638090, 1.5),
        (153.2088886, 128.5575220, 1.5),
    )
    if tuple(user.position_m for user in rt_settings.users) != expected_users:
        return False
    try:
        assets = resolve_builtin_scene_assets("ground_wall")
    except Exception:
        return False
    return (
        snapshot.metadata.get("scene_asset_sha256") == assets.file_sha256
        and snapshot.metadata.get("scene_bundle_sha256") == assets.bundle_sha256
    )


def _web_zero_orientations(rt_settings: RtBeamSettings) -> bool:
    return (
        rt_settings.receiver.orientation_deg == (0.0, 0.0, 0.0)
        and all(user.orientation_deg == (0.0, 0.0, 0.0) for user in rt_settings.users)
    )


def _web_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _web_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_web_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _web_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _web_json_safe(value.item())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, complex):
        return [
            _web_json_safe(value.real),
            _web_json_safe(value.imag),
        ]
    return value

def _web_values_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_web_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_web_values_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return bool(np.all(np.isfinite(value)))
        return all(_web_values_finite(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return _web_values_finite(value.item())
    if isinstance(value, bool) or isinstance(value, int):
        return True
    if isinstance(value, (float, complex)):
        return bool(np.isfinite(value))
    return True

def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    if denominator == 0.0:
        return 0.0 if float(np.linalg.norm(actual.ravel())) == 0.0 else math.inf
    return float(np.linalg.norm((actual - reference).ravel()) / denominator)


def validate_rt_los_snapshot(
    tx_settings: TxSettings,
    rt_settings: RtBeamSettings,
    snapshot: RtBeamSnapshot,
) -> dict[str, Any]:
    """Verify the direct RT path against geometry, Friis loss, and array phase."""
    rt_settings.validate_transmitter(tx_settings)
    if not rt_settings.rt.synthetic_array:
        raise ValueError("LoS analytic phase oracle requires synthetic_array=true")

    receiver = np.asarray(rt_settings.receiver.position_m, dtype=np.float64)
    wavelength = _SPEED_OF_LIGHT_M_S / rt_settings.rt.carrier_frequency_hz
    checks: list[dict[str, Any]] = []
    for user_index, user in enumerate(rt_settings.users):
        valid_paths = np.flatnonzero(
            np.any(snapshot.path_valid[:, user_index, :], axis=0)
        )
        direct_paths: list[int] = []
        for path_index in valid_paths:
            interactions, _ = _path_interactions_and_vertices(
                snapshot, user_index, int(path_index)
            )
            if not np.any(interactions):
                direct_paths.append(int(path_index))
        if len(direct_paths) != 1:
            checks.append(
                {
                    "check": "single_direct_path",
                    "user": user.name,
                    "path_count": len(direct_paths),
                    "passed": False,
                }
            )
            continue

        path_index = direct_paths[0]
        displacement = np.asarray(user.position_m, dtype=np.float64) - receiver
        distance = float(np.linalg.norm(displacement))
        direction = displacement / distance
        observed_distance = float(
            snapshot.path_tau_s[0, user_index, path_index] * _SPEED_OF_LIGHT_M_S
        )
        delay_relative_error = abs(observed_distance / distance - 1.0)
        steering = np.exp(
            2j * np.pi * (snapshot.element_positions_m @ direction) / wavelength
        )
        coefficients = snapshot.path_a[:, user_index, path_index]
        center_coefficient = np.mean(coefficients * steering.conj())
        reconstructed = center_coefficient * steering
        spatial_relative_error = _relative_error(reconstructed, coefficients)
        friis_amplitude = wavelength / (4.0 * np.pi * distance)
        friis_power_relative_error = abs(
            abs(center_coefficient) ** 2 / friis_amplitude**2 - 1.0
        )
        expected_coefficient = friis_amplitude * np.exp(
            -2j * np.pi * rt_settings.rt.carrier_frequency_hz
            * distance
            / _SPEED_OF_LIGHT_M_S
        )
        carrier_phase_relative_error = abs(
            center_coefficient / expected_coefficient - 1.0
        )
        passed = (
            delay_relative_error <= 1e-5
            and spatial_relative_error <= 1e-3
            and friis_power_relative_error <= 0.01
            and carrier_phase_relative_error <= 1e-3
        )
        checks.append(
            {
                "check": "direct_los_physics",
                "user": user.name,
                "path_index": path_index,
                "expected_distance_m": distance,
                "observed_distance_m": observed_distance,
                "delay_relative_error": delay_relative_error,
                "spatial_response_relative_error": spatial_relative_error,
                "friis_power_relative_error": friis_power_relative_error,
                "carrier_phase_relative_error": carrier_phase_relative_error,
                "passed": passed,
            }
        )
    return {
        "stage": "rt-los",
        "checks": checks,
        "path_count_per_user": snapshot.metadata["valid_path_count_per_user"],
        "passed": len(checks) == len(rt_settings.users)
        and all(check["passed"] for check in checks),
    }


def validate_rt_beam_structure(snapshot: RtBeamSnapshot) -> dict[str, Any]:
    """Verify shared-array combination identities and steering main lobes."""
    weights = np.asarray(snapshot.weights, dtype=np.complex128)
    positions = np.asarray(snapshot.element_positions_m, dtype=np.float64)
    if weights.shape != (positions.shape[0], 4):
        raise ValueError("RT beam weights must have shape [element,4]")
    h_ant = np.asarray(snapshot.h_ant, dtype=np.complex128)
    combined = combine_beams(h_ant, weights)
    combination_error = _relative_error(combined, snapshot.h_beam)

    gram = weights.conj().T @ weights
    gram_hermitian_error = _relative_error(gram, gram.conj().T)
    minimum_gram_eigenvalue = float(np.linalg.eigvalsh(gram).min())
    rng = np.random.default_rng(13)
    arbitrary_channel = (
        rng.standard_normal((weights.shape[0], 4))
        + 1j * rng.standard_normal((weights.shape[0], 4))
    ) / math.sqrt(2.0)
    symbols = (
        rng.standard_normal(4) + 1j * rng.standard_normal(4)
    ) / math.sqrt(2.0)
    matrix_identity_error = _relative_error(
        weights.conj().T @ (arbitrary_channel @ symbols),
        (weights.conj().T @ arbitrary_channel) @ symbols,
    )
    vector = (
        rng.standard_normal(weights.shape[0])
        + 1j * rng.standard_normal(weights.shape[0])
    ) / math.sqrt(2.0)
    projected_energy = float(np.linalg.norm(weights.conj().T @ vector) ** 2)
    projector_energy = float(
        np.real(np.vdot(vector, weights @ weights.conj().T @ vector))
    )
    energy_identity_relative_error = abs(projected_energy - projector_energy) / max(
        projected_energy, projector_energy, 1e-30
    )

    carrier_hz = float(snapshot.metadata["carrier_frequency_hz"])
    wavelength = _SPEED_OF_LIGHT_M_S / carrier_hz
    directions = np.asarray(snapshot.metadata["steering_directions_world"])
    rebuilt_weights = make_beam_weights(
        positions,
        directions,
        carrier_hz,
        phase_bits=int(snapshot.metadata["rt_config"]["beams"]["phase_bits"]),
    )
    weight_reconstruction_error = _relative_error(weights, rebuilt_weights)
    steering_gains: list[dict[str, Any]] = []
    for beam_index, direction in enumerate(directions):
        steering = np.exp(2j * np.pi * (positions @ direction) / wavelength)
        gains = np.abs(weights.conj().T @ steering)
        steering_gains.append(
            {
                "beam": beam_index,
                "matched_gain": float(gains[beam_index]),
                "strongest_other_gain": float(np.max(np.delete(gains, beam_index))),
                "passed": bool(gains[beam_index] >= np.max(np.delete(gains, beam_index))),
            }
        )
    checks = [
        {
            "check": "snapshot_beam_combination",
            "relative_error": combination_error,
            "threshold": 1e-12,
            "passed": combination_error <= 1e-12,
        },
        {
            "check": "gram_hermitian_positive_semidefinite",
            "hermitian_relative_error": gram_hermitian_error,
            "minimum_eigenvalue": minimum_gram_eigenvalue,
            "passed": gram_hermitian_error <= 1e-12
            and minimum_gram_eigenvalue >= -1e-12,
        },
        {
            "check": "beam_combination_linearity",
            "relative_error": matrix_identity_error,
            "threshold": 1e-12,
            "passed": matrix_identity_error <= 1e-12,
        },
        {
            "check": "projected_energy_identity",
            "relative_error": energy_identity_relative_error,
            "threshold": 1e-12,
            "passed": energy_identity_relative_error <= 1e-12,
        },
        {
            "check": "configured_steering_weights",
            "relative_error": weight_reconstruction_error,
            "threshold": 1e-12,
            "passed": weight_reconstruction_error <= 1e-12,
        },
        {
            "check": "per_beam_steering_main_lobe",
            "passed": all(item["passed"] for item in steering_gains),
        },
    ]
    return {
        "stage": "beam",
        "weight_column_norms": np.linalg.norm(weights, axis=0).tolist(),
        "weight_gram": {
            "real": gram.real.tolist(),
            "imag": gram.imag.tolist(),
        },
        "steering_gains": steering_gains,
        "channel_metrics": compute_beam_metrics(
            snapshot.h_beam, snapshot.power_scale_db
        ),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def validate_rt_noise_model(
    snapshot: RtBeamSnapshot,
    *,
    sample_count: int = 200_000,
    batch_size: int = 20_000,
) -> dict[str, Any]:
    """Empirically check correlated noise covariance and positive-definite whitening."""
    if sample_count < 1 or batch_size < 1:
        raise ValueError("noise sample_count and batch_size must be positive")
    weights = torch.as_tensor(snapshot.weights, dtype=torch.complex128, device="cpu")
    checks: list[dict[str, Any]] = []
    overall_passed = True
    ratios = (0.0, 0.001, 0.01)
    seeds = (13, 29, 47)
    for ratio in ratios:
        covariance = beam_noise_covariance(weights, 1.0, ratio).numpy()
        eigenvalues = np.linalg.eigvalsh(covariance)
        positive_definite = bool(eigenvalues[0] > max(eigenvalues[-1], 1.0) * 1e-12)
        cholesky = np.linalg.cholesky(covariance) if positive_definite else None
        for seed in seeds:
            empirical = np.zeros((4, 4), dtype=np.complex128)
            whitened_empirical = np.zeros_like(empirical)
            generated = 0
            batch_index = 0
            while generated < sample_count:
                count = min(batch_size, sample_count - generated)
                noise = add_correlated_awgn(
                    torch.zeros((count, 4), dtype=torch.complex128),
                    weights,
                    array_noise_variance=1.0,
                    post_noise_variance=ratio,
                    beam_axis=1,
                    seed=seed + batch_index * 104_729,
                ).cpu().numpy()
                empirical += noise.T @ noise.conj()
                if cholesky is not None:
                    whitened = np.linalg.solve(cholesky, noise.T).T
                    whitened_empirical += whitened.T @ whitened.conj()
                generated += count
                batch_index += 1
            empirical /= sample_count
            covariance_relative_error = _relative_error(empirical, covariance)
            whitening_relative_error = (
                _relative_error(
                    whitened_empirical / sample_count,
                    np.eye(4, dtype=np.complex128),
                )
                if cholesky is not None
                else None
            )
            passed = (
                covariance_relative_error <= 0.02
                and (
                    whitening_relative_error is None
                    or whitening_relative_error <= 0.02
                )
            )
            overall_passed = overall_passed and passed
            checks.append(
                {
                    "ratio": ratio,
                    "seed": seed,
                    "samples": sample_count,
                    "empirical_covariance_relative_error": covariance_relative_error,
                    "positive_definite": positive_definite,
                    "numerical_rank": int(np.linalg.matrix_rank(covariance)),
                    "whitened_covariance_relative_error": whitening_relative_error,
                    "passed": passed,
                }
            )

    singular_weights = weights.clone()
    singular_weights[:, 1] = singular_weights[:, 0]
    singular_cases: list[dict[str, Any]] = []
    for ratio, expected_rank in ((0.0, 3), (0.001, 4)):
        covariance = beam_noise_covariance(
            singular_weights, 1.0, ratio
        ).numpy()
        rank = int(np.linalg.matrix_rank(covariance))
        passed = rank == expected_rank
        overall_passed = overall_passed and passed
        singular_cases.append(
            {
                "ratio": ratio,
                "numerical_rank": rank,
                "expected_rank": expected_rank,
                "cholesky_allowed": rank == 4,
                "passed": passed,
            }
        )
    return {
        "stage": "noise",
        "sample_count_per_seed_ratio": sample_count,
        "ratios": list(ratios),
        "seeds": list(seeds),
        "checks": checks,
        "singular_covariance_checks": singular_cases,
        "passed": overall_passed,
    }


def validate_rt_uncoded(snapshot: RtBeamSnapshot, *, seed: int = 13) -> dict[str, Any]:
    """Check noiseless ZF, independent residual power, and isolated QPSK BER."""
    powers = user_power_scales(snapshot.power_scale_db)
    channel_by_bin = (
        snapshot.h_beam
        * np.sqrt(powers)[None, :, None]
    ).transpose(2, 0, 1)
    singular_values = np.linalg.svd(channel_by_bin, compute_uv=False)
    conditioning = singular_values[:, -1] / singular_values[:, 0]
    frequency_bin = int(np.argmax(conditioning))
    channel = np.asarray(channel_by_bin[frequency_bin], dtype=np.complex128)
    rng = np.random.default_rng(seed)
    sample_count = 200_000
    bit_pairs = rng.integers(0, 2, size=(4, sample_count, 2), dtype=np.int8)
    symbols = (
        (1.0 - 2.0 * bit_pairs[:, :, 0])
        + 1j * (1.0 - 2.0 * bit_pairs[:, :, 1])
    ) / math.sqrt(2.0)
    received = channel @ symbols
    matrix_rank = int(np.linalg.matrix_rank(channel))
    if matrix_rank == 4:
        zf_symbols = np.linalg.solve(channel, received)
        zf_relative_error = _relative_error(zf_symbols, symbols)
    else:
        zf_relative_error = math.inf

    independent_residuals: list[dict[str, Any]] = []
    for user in range(4):
        desired = channel[user, user]
        if abs(desired) == 0.0:
            independent_residuals.append(
                {
                    "user": user,
                    "expected_residual_power": None,
                    "observed_residual_power": None,
                    "relative_error": math.inf,
                    "passed": False,
                }
            )
            continue
        estimate = received[user] / desired
        residual = estimate - symbols[user]
        expected = float(
            np.sum(np.abs(channel[user, np.arange(4) != user]) ** 2)
            / abs(desired) ** 2
        )
        observed = float(np.mean(np.abs(residual) ** 2))
        relative_error = (
            abs(observed / expected - 1.0)
            if expected > 0.0
            else (0.0 if observed <= 1e-12 else math.inf)
        )
        independent_residuals.append(
            {
                "user": user,
                "expected_residual_power": expected,
                "observed_residual_power": observed,
                "relative_error": relative_error,
                "passed": relative_error <= 0.03,
            }
        )

    es_n0_db = 6.0
    es_n0 = 10.0 ** (es_n0_db / 10.0)
    qpsk_bits = rng.integers(0, 2, size=(sample_count, 2), dtype=np.int8)
    qpsk_symbols = (
        (1.0 - 2.0 * qpsk_bits[:, 0])
        + 1j * (1.0 - 2.0 * qpsk_bits[:, 1])
    ) / math.sqrt(2.0)
    qpsk_noise_variance = abs(channel[0, 0]) ** 2 / es_n0
    qpsk_noise = math.sqrt(qpsk_noise_variance / 2.0) * (
        rng.standard_normal(sample_count) + 1j * rng.standard_normal(sample_count)
    )
    qpsk_received = channel[0, 0] * qpsk_symbols + qpsk_noise
    qpsk_equalized = qpsk_received * channel[0, 0].conjugate() / abs(channel[0, 0]) ** 2
    detected_bits = np.stack(
        [qpsk_equalized.real < 0.0, qpsk_equalized.imag < 0.0], axis=1
    )
    bit_errors = int(np.count_nonzero(detected_bits != qpsk_bits))
    bit_count = int(qpsk_bits.size)
    measured_ber = bit_errors / bit_count
    theoretical_ber = 0.5 * math.erfc(math.sqrt(es_n0 / 2.0))
    confidence = _wilson_interval(bit_errors, bit_count)
    qpsk_passed = confidence[0] <= theoretical_ber <= confidence[1]
    checks = [
        {
            "check": "noiseless_full_rank_zf",
            "matrix_rank": matrix_rank,
            "relative_error": zf_relative_error,
            "threshold": 1e-9,
            "passed": matrix_rank == 4 and zf_relative_error <= 1e-9,
        },
        {
            "check": "beam_independent_residual_power",
            "users": independent_residuals,
            "threshold": 0.03,
            "passed": all(item["passed"] for item in independent_residuals),
        },
        {
            "check": "isolated_qpsk_awgn_ber",
            "es_n0_db": es_n0_db,
            "bit_errors": bit_errors,
            "bits": bit_count,
            "measured_ber": measured_ber,
            "theoretical_ber": theoretical_ber,
            "wilson_95_interval": list(confidence),
            "passed": qpsk_passed,
        },
    ]
    return {
        "stage": "uncoded",
        "frequency_bin": frequency_bin,
        "condition_ratio": float(conditioning[frequency_bin]),
        "sample_count": sample_count,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def _wilson_interval(errors: int, trials: int) -> tuple[float, float]:
    if trials < 1 or not 0 <= errors <= trials:
        raise ValueError("Wilson interval requires 0 <= errors <= positive trials")
    z = 1.959_963_984_540_054
    estimate = errors / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def run_rt_beam_smoke_sweeps(
    tx_settings: TxSettings,
    rt_settings: RtBeamSettings,
    simulation_settings: BlerSettings,
    *,
    device: str | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run the planned one-factor RT/PUSCH scans with a fixed two-frame budget."""
    tx_settings.validate()
    rt_settings.validate_transmitter(tx_settings)
    simulation_settings.validate()
    if simulation_settings.max_frames_per_snr != 2:
        raise ValueError("RT beam smoke sweeps require max_frames_per_snr=2")
    if simulation_settings.target_block_errors < 8:
        raise ValueError("RT beam smoke sweeps require target_block_errors>=8")
    if simulation_settings.stop_at_zero_bler:
        raise ValueError("RT beam smoke sweeps require stop_at_zero_bler=false")
    if "dmrs-lmmse" in simulation_settings.channel_estimators_for_sweep:
        raise ValueError("RT beam smoke sweeps do not support the CDL dmrs-lmmse prior")

    detector_parameters = {
        name: value
        for name, value in simulation_settings.detector_parameters.items()
        if name not in {"zf", "beam-independent"}
    }
    sweep_settings = replace(
        simulation_settings,
        channel_estimator="perfect",
        channel_estimators=("perfect", "dmrs"),
        detector="lmmse",
        detectors=("beam-independent", "zf", "lmmse"),
        detector_parameter=None,
        detector_parameters=detector_parameters,
    )
    sweep_settings.validate()

    variants: list[tuple[str, Any, Any]] = []
    receiver = rt_settings.receiver.position_m
    angle_multipliers = (-1.5, -0.5, 0.5, 1.5)
    for spacing_deg in (5.0, 10.0, 20.0, 30.0):
        users = tuple(
            replace(
                user,
                position_m=(
                    receiver[0] + 200.0 * math.cos(
                        math.radians(angle_multipliers[index] * spacing_deg)
                    ),
                    receiver[1] + 200.0 * math.sin(
                        math.radians(angle_multipliers[index] * spacing_deg)
                    ),
                    user.position_m[2],
                ),
            )
            for index, user in enumerate(rt_settings.users)
        )
        variants.append(
            (f"azimuth_spacing_deg={spacing_deg:g}", replace(rt_settings, users=users), {
                "factor": "user_azimuth_spacing_deg",
                "value": spacing_deg,
            })
        )

    for power_db in (0.0, 10.0, 20.0, 30.0):
        users = tuple(
            replace(user, power_scale_db=power_db) if index == 3 else user
            for index, user in enumerate(rt_settings.users)
        )
        variants.append(
            (f"ue3_power_scale_db={power_db:g}", replace(rt_settings, users=users), {
                "factor": "ue3_power_scale_db",
                "value": power_db,
            })
        )

    for offset_deg in (0.0, 1.0, 2.0, 4.0):
        beams = replace(
            rt_settings.beams,
            azimuth_offset_deg=(offset_deg,) * 4,
        )
        variants.append(
            (f"common_beam_azimuth_offset_deg={offset_deg:g}", replace(rt_settings, beams=beams), {
                "factor": "common_beam_azimuth_offset_deg",
                "value": offset_deg,
            })
        )

    for phase_bits in (0, 2, 3, 4, 6):
        beams = replace(rt_settings.beams, phase_bits=phase_bits)
        variants.append(
            (f"phase_bits={phase_bits}", replace(rt_settings, beams=beams), {
                "factor": "phase_bits",
                "value": phase_bits,
            })
        )

    for scene, max_depth in (("empty", 0), ("ground", 1), ("ground_wall", 2)):
        scene_settings = replace(
            rt_settings.rt,
            scene=scene,
            max_depth=max_depth,
        )
        variants.append(
            (f"scene={scene}", replace(rt_settings, rt=scene_settings), {
                "factor": "scene",
                "value": scene,
            })
        )

    points: list[dict[str, Any]] = []
    passed = True
    total = len(variants)
    for index, (name, resolved_rt, factor) in enumerate(variants, start=1):
        snapshot = prepare_rt_beam_snapshot(tx_settings, resolved_rt)
        rows = simulate_rt_beam_bler(
            tx_settings,
            snapshot,
            sweep_settings,
            post_combiner_ratio=resolved_rt.noise.post_combiner_ratio,
            device=device,
        )
        status_counts: dict[str, int] = {}
        for row in rows:
            status_counts[row.status] = status_counts.get(row.status, 0) + 1
        point_passed = all(
            row.status in {"complete", "infeasible_rank"} for row in rows
        ) and len(rows) > 0
        passed = passed and point_passed
        points.append(
            {
                "name": name,
                **factor,
                "resolved_rt_config": resolved_rt.to_dict(),
                "resolved_rt_toml": resolved_rt.to_toml(),
                "snapshot_config_sha256": snapshot.metadata["config_sha256"],
                "snapshot_solver": snapshot.metadata["solver"],
                "status_counts": status_counts,
                "rows": [asdict(row) for row in rows],
                "passed": point_passed,
            }
        )
        if on_progress is not None:
            on_progress(index, total, name)
    return {
        "stage": "sweeps",
        "point_count": len(points),
        "frame_budget_per_snr": 2,
        "estimators": list(sweep_settings.channel_estimators_for_sweep),
        "detectors": list(sweep_settings.detectors),
        "points": points,
        "passed": passed and len(points) == total,
    }


def _bler_settings_toml(settings: BlerSettings) -> str:
    lines = ["[bler]"]
    for key, value in settings.to_dict().items():
        if value is None:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


