"""Coded PUSCH simulation over a fixed RT snapshot and shared receive beams."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import scipy.stats
import sionna.phy
import torch

from .beamforming import user_power_scales
from .config import TxSettings
from .device import use_device
from .noise import add_correlated_awgn, beam_noise_covariance
from .receiver import NrPuschRx
from .rt_channel import NrPuschRtBeamChannel, RtBeamSnapshot
from .simulation_config import BlerSettings
from .transmitter import NrPuschTx


@dataclass(frozen=True)
class BeamBlerRow:
    channel_estimator: str
    detector: str
    device: str
    snr_db: float
    seed: int
    post_combiner_ratio: float
    user: str
    frames: int
    transport_blocks: int
    block_errors: int
    crc_failures: int
    bit_errors: int
    bits: int
    bler: float | None
    bler_ci95_low: float | None
    bler_ci95_high: float | None
    crc_fail_rate: float | None
    ber: float | None
    runtime_s: float
    reference_snr_db: float
    actual_snr_db: float | None
    observation_basis: str
    status: str
    reason: str | None


def simulate_rt_beam_bler(
    tx_settings: TxSettings,
    snapshot: RtBeamSnapshot,
    simulation_settings: BlerSettings,
    *,
    post_combiner_ratio: float,
    seed: int | None = None,
    device: str | None = None,
    on_point: Callable[[tuple[BeamBlerRow, ...]], None] | None = None,
) -> tuple[BeamBlerRow, ...]:
    """Run coded PUSCH arms against one deterministic static RT realization."""
    tx_settings.validate()
    simulation_settings.validate()
    ratio = _validate_ratio(post_combiner_ratio)
    effective_seed = simulation_settings.seed if seed is None else _validate_seed(seed)
    settings = simulation_settings
    if seed is not None:
        settings = replace(simulation_settings, seed=effective_seed)
        settings.validate()
    if len(tx_settings.users) != 4 or tx_settings.pusch.num_layers != 1 or tx_settings.pusch.num_antenna_ports != 1:
        raise ValueError("RT beam BLER 要求四用户、每用户单层和单发射天线")
    if [user.name for user in tx_settings.users] != snapshot.metadata.get("users"):
        raise ValueError("RT snapshot UE 顺序与 TX 配置不一致")
    estimator_arms = settings.channel_estimators_for_sweep
    if "dmrs-lmmse" in estimator_arms:
        raise ValueError("RT beam channel 不支持 CDL dmrs-lmmse prior；请使用 dmrs 或 perfect")
    receiver_config = snapshot.metadata["rt_config"]["receiver"]
    snapshot_l_min = int(snapshot.metadata["l_min"])
    snapshot_max_delay = float(snapshot.metadata["max_delay_spread_s"])
    if settings.l_min != snapshot_l_min:
        raise ValueError(
            f"BlerSettings.l_min={settings.l_min} 与 RT snapshot l_min={snapshot_l_min} 不一致"
        )
    max_delay_spread_s = (
        snapshot_max_delay if settings.max_delay_spread_s is None
        else float(settings.max_delay_spread_s)
    )
    if max_delay_spread_s != snapshot_max_delay:
        raise ValueError(
            "BlerSettings.max_delay_spread_s 与 RT snapshot max_delay_spread_s 不一致"
        )
    if int(receiver_config["l_min"]) != snapshot_l_min:
        raise ValueError("RT snapshot receiver l_min 元数据不一致")
    if float(receiver_config["max_delay_spread_s"]) != snapshot_max_delay:
        raise ValueError("RT snapshot receiver max_delay_spread_s 元数据不一致")
    if settings.channel_domain == "frequency" and not bool(snapshot.metadata.get("cp_sufficient", False)):
        raise ValueError("RT snapshot CP 不充分；请使用 channel_domain='time'，不能声称频域等价")
    power_db = np.asarray(snapshot.power_scale_db, dtype=np.float64)
    if power_db.shape != (4,) or power_db[0] != 0.0:
        raise ValueError("RT beam BLER 要求 UE0 power_scale_db 固定为 0 dB")

    device_name = use_device(device or settings.device)
    sionna.phy.config.seed = effective_seed
    torch.manual_seed(effective_seed)
    transmitter = NrPuschTx(tx_settings, device=device_name)
    resource_grid = transmitter._tx_freq.resource_grid
    if int(resource_grid.fft_size) != int(snapshot.metadata["fft_size"]):
        raise ValueError("RT snapshot FFT size 与 TX resource grid 不一致")
    if int(resource_grid.num_ofdm_symbols) != int(snapshot.metadata["num_ofdm_symbols"]):
        raise ValueError("RT snapshot OFDM symbol count 与 TX resource grid 不一致")
    if transmitter.sample_rate_hz != int(snapshot.metadata["sample_rate_hz"]):
        raise ValueError("RT snapshot sample rate 与 TX 配置不一致")
    channel = NrPuschRtBeamChannel(snapshot, device=device_name)
    weights = torch.as_tensor(snapshot.weights, dtype=torch.complex64, device=device_name)
    data_bins = _data_subcarrier_indices(transmitter)
    raw_h_beam = torch.as_tensor(snapshot.h_beam, dtype=torch.complex64, device=device_name)
    powers = torch.as_tensor(
        user_power_scales(power_db), dtype=torch.float32, device=device_name
    )
    effective_channel = raw_h_beam * powers.sqrt()[None, :, None]
    reference_power = float(
        (effective_channel[0, 0].index_select(0, data_bins).abs().square().mean()).item()
    )
    if not math.isfinite(reference_power) or reference_power <= 0.0:
        raise ValueError("RT snapshot UE0 reference data-RE channel power is zero or non-finite")
    array_gain = float(weights[:, 0].abs().square().sum().item())
    if not math.isfinite(array_gain) or array_gain <= 0.0:
        raise ValueError("UE0 beam has zero or non-finite array gain")

    rank_failure = _zf_rank_failure(effective_channel, data_bins)
    rows: list[BeamBlerRow] = []
    effective_batch_size = settings.batch_size
    for estimator in estimator_arms:
        for detector in settings.detectors:
            parameter = settings.detector_parameters.get(
                detector, settings.detector_parameter
            )
            if detector in {"zf", "beam-independent"} and parameter is not None:
                raise ValueError(f"{detector} 不接受 detector_parameter")
            receiver = NrPuschRx(
                tx_settings,
                channel_estimator=estimator,
                l_min=snapshot_l_min,
                max_delay_spread_s=max_delay_spread_s,
                num_decoder_iterations=settings.num_decoder_iterations,
                detector=detector,
                detector_parameter=parameter,
                detector_damping=settings.detector_damping,
                input_domain=settings.channel_domain,
                device=device_name,
            )
            observation_basis = (
                "diagonal_noise_scaled" if detector == "beam-independent" else "whitened"
            )
            joint_covariance_status = detector != "beam-independent"
            stop_trigger_snr_db: float | None = None
            for snr_db in settings.snr_db:
                if stop_trigger_snr_db is not None:
                    point_rows = _skipped_rows(
                        estimator=estimator,
                        detector=detector,
                        device=device_name,
                        snr_db=float(snr_db),
                        seed=effective_seed,
                        ratio=ratio,
                        tx_settings=tx_settings,
                        reference_snr_db=float(snr_db),
                        observation_basis=observation_basis,
                        trigger_snr_db=stop_trigger_snr_db,
                    )
                    rows.extend(point_rows)
                    if on_point is not None:
                        on_point(point_rows)
                    continue
                started = time.perf_counter()
                array_variance, post_variance, covariance, chol = _noise_context(
                    reference_power,
                    array_gain,
                    float(snr_db),
                    ratio,
                    weights,
                )
                covariance_is_singular = chol is None
                status = "complete"
                reason = None
                if detector == "zf" and rank_failure is not None:
                    status = "infeasible_rank"
                    reason = rank_failure
                elif joint_covariance_status and covariance_is_singular:
                    status = "singular_noise_covariance"
                    reason = "R_eta is positive semidefinite but singular; joint whitening is undefined"
                if status != "complete":
                    point_rows = _failure_rows(
                        estimator, detector, device_name, float(snr_db), effective_seed,
                        ratio, tx_settings, effective_channel, covariance,
                        data_bins, observation_basis, status, reason,
                    )
                    rows.extend(point_rows)
                    if on_point is not None:
                        on_point(point_rows)
                    continue

                per_user_block_errors = np.zeros(4, dtype=np.int64)
                per_user_crc_failures = np.zeros(4, dtype=np.int64)
                per_user_bit_errors = np.zeros(4, dtype=np.int64)
                frames = 0
                while (
                    frames < settings.max_frames_per_snr
                    and int(per_user_block_errors.sum()) < settings.target_block_errors
                ):
                    batch_size = min(
                        effective_batch_size,
                        settings.max_frames_per_snr - frames,
                    )
                    tx_result = transmitter.generate(
                        batch_size=batch_size,
                        seed=_payload_seed(effective_seed, frames),
                    )
                    if settings.channel_domain == "frequency":
                        channel_result = channel.apply_frequency(
                            tx_result.frequency_grid, resource_grid
                        )
                        observed = add_correlated_awgn(
                            channel_result.grid[:, 0],
                            weights,
                            array_noise_variance=array_variance,
                            post_noise_variance=post_variance,
                            beam_axis=1,
                            seed=_noise_seed(effective_seed, frames),
                        ).unsqueeze(1)
                        perfect_csi = channel_result.channel_frequency_response
                        if detector == "beam-independent":
                            observed = _diagonal_scale(observed, covariance, beam_axis=2)
                            perfect_csi = _diagonal_scale(
                                perfect_csi, covariance, beam_axis=2
                            )
                        else:
                            assert chol is not None
                            observed = _whiten(observed, chol, beam_axis=2)
                            perfect_csi = _whiten(perfect_csi, chol, beam_axis=2)
                        result = receiver.receive_frequency_grid(
                            observed,
                            1.0,
                            channel_frequency_response=(
                                perfect_csi if estimator == "perfect" else None
                            ),
                        )
                    else:
                        channel_result = channel.apply(tx_result.iq, tx_result.sample_rate_hz)
                        observed = add_correlated_awgn(
                            channel_result.iq,
                            weights,
                            array_noise_variance=array_variance,
                            post_noise_variance=post_variance,
                            beam_axis=1,
                            seed=_noise_seed(effective_seed, frames),
                        )
                        perfect_taps = channel_result.channel_taps
                        if detector == "beam-independent":
                            observed = _diagonal_scale(observed, covariance, beam_axis=1)
                            perfect_taps = _diagonal_scale(
                                perfect_taps, covariance, beam_axis=2
                            )
                        else:
                            assert chol is not None
                            observed = _whiten(observed, chol, beam_axis=1)
                            perfect_taps = _whiten(perfect_taps, chol, beam_axis=2)
                        result = receiver.receive(
                            observed,
                            1.0,
                            channel_taps=(perfect_taps if estimator == "perfect" else None),
                        )
                    if result.bits.shape != tx_result.bits.shape or result.crc_status.shape != tx_result.bits.shape[:2]:
                        raise RuntimeError("RT beam RX bits/CRC 轴与 TX payload 不一致")
                    errors = torch.any(result.bits != tx_result.bits, dim=-1)
                    block_errors = torch.logical_or(~result.crc_status, errors)
                    per_user_block_errors += block_errors.sum(dim=0).detach().cpu().numpy()
                    per_user_crc_failures += (~result.crc_status).sum(dim=0).detach().cpu().numpy()
                    per_user_bit_errors += (
                        (result.bits != tx_result.bits).sum(dim=(0, 2)).detach().cpu().numpy()
                    )
                    frames += batch_size

                runtime_s = time.perf_counter() - started
                point_rows = _complete_rows(
                    estimator=estimator,
                    detector=detector,
                    device=device_name,
                    snr_db=float(snr_db),
                    seed=effective_seed,
                    ratio=ratio,
                    tx_settings=tx_settings,
                    tbs=transmitter.transport_block_size,
                    frames=frames,
                    block_errors=per_user_block_errors,
                    crc_failures=per_user_crc_failures,
                    bit_errors=per_user_bit_errors,
                    runtime_s=runtime_s,
                    reference_snr_db=float(snr_db),
                    actual_snr_db=_actual_snr_db(effective_channel, covariance, data_bins),
                    observation_basis=observation_basis,
                )
                rows.extend(point_rows)
                if on_point is not None:
                    on_point(point_rows)
                if settings.stop_at_zero_bler and point_rows[-1].block_errors == 0:
                    stop_trigger_snr_db = float(snr_db)
            del receiver
    return tuple(rows)


def save_rt_beam_bler(
    rows: tuple[BeamBlerRow, ...],
    output: str | Path,
    *,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Write per-user/aggregate CSV rows and their complete JSON manifest."""
    output_path = Path(output)
    if output_path.suffix.lower() != ".csv":
        raise ValueError("RT beam BLER 输出路径必须以 .csv 结尾")
    if not rows:
        raise ValueError("RT beam BLER 至少需要一个结果行")
    if not isinstance(metadata, dict):
        raise ValueError("RT beam BLER manifest metadata 必须是映射")
    for row in rows:
        if row.status not in {
            "complete", "infeasible_rank", "singular_noise_covariance", "skipped",
        }:
            raise ValueError(f"不支持的 RT beam BLER status: {row.status}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".json")
    fieldnames = list(asdict(rows[0]))
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    manifest = dict(metadata)
    manifest["schema_version"] = 1
    manifest["channel_sampling"] = "fixed_rt_snapshot"
    manifest["result"] = {
        "csv": output_path.name,
        "row_count": len(rows),
        "status_counts": {
            status: sum(row.status == status for row in rows)
            for status in ("complete", "infeasible_rank", "singular_noise_covariance", "skipped")
        },
        "skipped_points": [
            {
                "channel_estimator": row.channel_estimator,
                "detector": row.detector,
                "snr_db": row.snr_db,
                "reason": row.reason,
            }
            for row in rows
            if row.user == "all" and row.status == "skipped"
        ],
        "skip_policy": (
            "stop_at_zero_bler=true skips higher SNRs independently per estimator/detector after an observed all-UE zero-error point"
            if any(row.status == "skipped" for row in rows)
            else "no RT SNR points were skipped"
        ),
        "bler_error_definition": "TB CRC failure OR decoded payload differs from transmitted payload",
        "aggregate_bler_definition": "unweighted mean of four per-user BLER estimates",
        "aggregate_ci95_definition": (
            "mean of per-user exact two-sided 98.75% Clopper-Pearson endpoints; "
            "Bonferroni simultaneous coverage is at least 95%"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path, manifest_path


def make_rt_beam_manifest_metadata(
    tx_settings: TxSettings,
    snapshot: RtBeamSnapshot,
    simulation_settings: BlerSettings,
    *,
    post_combiner_ratio: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Build reproducibility metadata, including every configured R_eta matrix."""
    ratio = _validate_ratio(post_combiner_ratio)
    effective_seed = _validate_seed(seed)
    transmitter = NrPuschTx(tx_settings, device=device)
    data_bins = _data_subcarrier_indices(transmitter)
    powers = np.asarray(user_power_scales(snapshot.power_scale_db), dtype=np.float64)
    h_beam = np.asarray(snapshot.h_beam)
    q_ref = float(np.mean(np.abs(h_beam[0, 0, data_bins.cpu().numpy()] * math.sqrt(powers[0])) ** 2))
    weights = torch.as_tensor(snapshot.weights, dtype=torch.complex128)
    w0_norm = float(torch.sum(torch.abs(weights[:, 0]) ** 2).item())
    covariance: dict[str, Any] = {}
    for snr_db in simulation_settings.snr_db:
        array_variance = _array_noise_variance(q_ref, w0_norm, float(snr_db), ratio)
        post_variance = ratio * array_variance
        matrix = beam_noise_covariance(weights, array_variance, post_variance).cpu().numpy()
        covariance[f"{float(snr_db):g}"] = {
            "real": matrix.real.tolist(),
            "imag": matrix.imag.tolist(),
            "array_noise_variance": array_variance,
            "post_noise_variance": post_variance,
        }
    return {
        "tx_settings": tx_settings.to_dict(),
        "simulation_settings": simulation_settings.to_dict(),
        "channel_snapshot_config_sha256": snapshot.metadata.get("config_sha256"),
        "channel_snapshot_scene_file": snapshot.metadata.get("scene_file"),
        "channel_snapshot_scene_source": snapshot.metadata.get("scene_source"),
        "channel_snapshot_scene_asset_sha256": snapshot.metadata.get("scene_asset_sha256"),
        "channel_snapshot_scene_bundle_sha256": snapshot.metadata.get("scene_bundle_sha256"),
        "scene_bundle_sha256": snapshot.metadata.get("scene_bundle_sha256"),
        "rt_settings": snapshot.metadata.get("rt_config"),
        "channel_snapshot_array_sha256": snapshot.metadata.get("array_sha256") or _snapshot_hash(snapshot),
        "channel_snapshot_scene": snapshot.metadata.get("scene"),
        "channel_snapshot_users": snapshot.metadata.get("users"),
        "effective_seed": effective_seed,
        "post_combiner_ratio": ratio,
        "reference_channel_power_q_ref": q_ref,
        "noise_covariance_by_snr_db": covariance,
        "observation_bases": {
            "beam-independent": "diagonal_noise_scaled",
            "joint_detectors": "whitened",
        },
        "physical_noise_model": "sigma_a^2 W^H W + sigma_v^2 I_4",
        "frequency_domain_assumption": "static per-RE channel; CP sufficiency is snapshot-gated",
        "cp_sufficient": bool(snapshot.metadata.get("cp_sufficient", False)),
        "actual_snr_db_definition": "10*log10(P_j*mean_data_RE(|H_beam[j,j]|^2)/R_eta[j,j])",
        "channel_hash_semantics": "fixed conditional RT realization; no per-frame channel resampling",
        "device": device,
    }


def save_rt_beam_diagnostic_captures(
    tx_settings: TxSettings,
    snapshot: RtBeamSnapshot,
    simulation_settings: BlerSettings,
    output_dir: str | Path,
    *,
    post_combiner_ratio: float,
    seed: int,
    device: str,
) -> dict[str, str | None]:
    """Save one reproducible noisy frame in physical, diagonal, and whitened bases."""
    ratio = _validate_ratio(post_combiner_ratio)
    effective_seed = _validate_seed(seed)
    simulation_settings.validate()
    tx_settings.validate()
    if len(tx_settings.users) != 4 or tx_settings.pusch.num_layers != 1 or tx_settings.pusch.num_antenna_ports != 1:
        raise ValueError("RT diagnostic capture 要求四用户、每用户单层和单发射天线")
    if [user.name for user in tx_settings.users] != snapshot.metadata.get("users"):
        raise ValueError("RT diagnostic capture UE 顺序与 snapshot 不一致")
    device_name = use_device(device)
    transmitter = NrPuschTx(tx_settings, device=device_name)
    resource_grid = transmitter._tx_freq.resource_grid
    if int(resource_grid.fft_size) != int(snapshot.metadata["fft_size"]):
        raise ValueError("RT diagnostic capture FFT size 与 snapshot 不一致")
    channel = NrPuschRtBeamChannel(snapshot, device=device_name)
    weights = torch.as_tensor(snapshot.weights, dtype=torch.complex64, device=device_name)
    data_bins = _data_subcarrier_indices(transmitter)
    powers = torch.as_tensor(
        user_power_scales(snapshot.power_scale_db), dtype=torch.float32, device=device_name
    )
    h_beam = torch.as_tensor(snapshot.h_beam, dtype=torch.complex64, device=device_name)
    effective_channel = h_beam * powers.sqrt()[None, :, None]
    reference_power = float(
        effective_channel[0, 0].index_select(0, data_bins).abs().square().mean().item()
    )
    array_gain = float(weights[:, 0].abs().square().sum().item())
    array_variance, post_variance, covariance, cholesky = _noise_context(
        reference_power,
        array_gain,
        float(simulation_settings.snr_db[0]),
        ratio,
        weights,
    )
    tx_result = transmitter.generate(batch_size=1, seed=_payload_seed(effective_seed, 0))
    channel_result = channel.apply_frequency(tx_result.frequency_grid, resource_grid)
    noisy_physical = add_correlated_awgn(
        channel_result.grid[:, 0],
        weights,
        array_noise_variance=array_variance,
        post_noise_variance=post_variance,
        beam_axis=1,
        seed=_noise_seed(effective_seed, 0),
    ).unsqueeze(1)
    physical_csi = channel_result.channel_frequency_response
    captures: dict[str, tuple[torch.Tensor, torch.Tensor, str | None]] = {
        "physical_beam": (noisy_physical, physical_csi, None),
        "diagonal_noise_scaled": (
            _diagonal_scale(noisy_physical, covariance, beam_axis=2),
            _diagonal_scale(physical_csi, covariance, beam_axis=2),
            "beam-independent",
        ),
    }
    whitening_status = "available"
    if cholesky is None:
        whitening_status = "unavailable_singular_noise_covariance"
    else:
        captures["whitened"] = (
            _whiten(noisy_physical, cholesky, beam_axis=2),
            _whiten(physical_csi, cholesky, beam_axis=2),
            "lmmse",
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    covariance_array = covariance.detach().cpu().numpy()
    covariance_json = {
        "real": covariance_array.real.tolist(),
        "imag": covariance_array.imag.tolist(),
    }
    paths: dict[str, str | None] = {}
    for basis, (grid, csi, detector) in captures.items():
        archive_path = destination / f"{basis}.npz"
        manifest_path = destination / f"{basis}.json"
        np.savez_compressed(
            archive_path,
            grid=grid.detach().cpu().numpy(),
            channel_frequency_response=csi.detach().cpu().numpy(),
            noiseless_grid=channel_result.grid.detach().cpu().numpy(),
            bits=tx_result.bits.detach().cpu().numpy().astype(np.uint8, copy=False),
            noise_covariance=covariance_array,
            weights=weights.detach().cpu().numpy(),
        )
        manifest = {
            "format_version": 1,
            "observation_basis": basis,
            "input_domain": "frequency",
            "channel_estimator": "perfect",
            "detector": detector,
            "noise_variance_for_replay": 1.0 if detector is not None else None,
            "snr_db": float(simulation_settings.snr_db[0]),
            "seed": effective_seed,
            "payload_seed": _payload_seed(effective_seed, 0),
            "noise_seed": _noise_seed(effective_seed, 0),
            "post_combiner_ratio": ratio,
            "array_noise_variance": array_variance,
            "post_noise_variance": post_variance,
            "noise_covariance": covariance_json,
            "grid_axes": ["batch", "stream", "rx_beam", "ofdm_symbol", "fft_bin"],
            "channel_axes": [
                "batch", "stream", "rx_beam", "user", "tx_antenna",
                "ofdm_symbol", "fft_bin",
            ],
            "bits_axes": ["batch", "user", "transport_block_bit"],
            "snapshot_config_sha256": snapshot.metadata.get("config_sha256"),
            "snapshot_array_sha256": snapshot.metadata.get("array_sha256") or _snapshot_hash(snapshot),
            "npz": archive_path.name,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths[basis] = str(archive_path)
    paths["whitening_status"] = whitening_status
    if "whitened" not in captures:
        paths["whitened"] = None
    return paths


def _complete_rows(
    *,
    estimator: str,
    detector: str,
    device: str,
    snr_db: float,
    seed: int,
    ratio: float,
    tx_settings: TxSettings,
    tbs: int,
    frames: int,
    block_errors: np.ndarray,
    crc_failures: np.ndarray,
    bit_errors: np.ndarray,
    runtime_s: float,
    reference_snr_db: float,
    actual_snr_db: list[float | None],
    observation_basis: str,
) -> tuple[BeamBlerRow, ...]:
    if frames < 1:
        raise RuntimeError("RT beam BLER produced no frames for a complete result")
    user_rows: list[BeamBlerRow] = []
    for index, tx_user in enumerate(tx_settings.users):
        errors = int(block_errors[index])
        lo, hi = _clopper_pearson(errors, frames, confidence=0.95)
        bler = errors / frames
        user_rows.append(
            BeamBlerRow(
                estimator, detector, device, snr_db, seed, ratio, tx_user.name,
                frames, frames, errors, int(crc_failures[index]), int(bit_errors[index]),
                frames * tbs, bler, lo, hi, int(crc_failures[index]) / frames,
                int(bit_errors[index]) / (frames * tbs), runtime_s, reference_snr_db,
                actual_snr_db[index], observation_basis, "complete", None,
            )
        )
    aggregate_low = sum(bounds[0] for bounds in (
        _clopper_pearson(int(block_errors[index]), frames, confidence=0.9875)
        for index in range(4)
    )) / 4.0
    aggregate_high = sum(bounds[1] for bounds in (
        _clopper_pearson(int(block_errors[index]), frames, confidence=0.9875)
        for index in range(4)
    )) / 4.0
    total_errors = int(block_errors.sum())
    total_crc_failures = int(crc_failures.sum())
    total_bit_errors = int(bit_errors.sum())
    total_tbs = 4 * frames
    total_bits = total_tbs * tbs
    aggregate = BeamBlerRow(
        estimator, detector, device, snr_db, seed, ratio, "all", frames, total_tbs,
        total_errors, total_crc_failures, total_bit_errors, total_bits,
        total_errors / total_tbs, aggregate_low, aggregate_high,
        total_crc_failures / total_tbs, total_bit_errors / total_bits,
        runtime_s, reference_snr_db, None, observation_basis, "complete", None,
    )
    return (*user_rows, aggregate)


def _failure_rows(
    estimator: str,
    detector: str,
    device: str,
    snr_db: float,
    seed: int,
    ratio: float,
    tx_settings: TxSettings,
    effective_channel: torch.Tensor,
    covariance: torch.Tensor,
    data_bins: torch.Tensor,
    observation_basis: str,
    status: str,
    reason: str,
) -> tuple[BeamBlerRow, ...]:
    actual_snr = _actual_snr_db(effective_channel, covariance, data_bins)
    users = [user.name for user in tx_settings.users]
    rows: list[BeamBlerRow] = []
    for index, user_name in enumerate(users):
        rows.append(
            BeamBlerRow(
                estimator, detector, device, snr_db, seed, ratio, user_name,
                0, 0, 0, 0, 0, 0, None, None, None, None, None, 0.0,
                snr_db, actual_snr[index], observation_basis, status, reason,
            )
        )
    rows.append(
        BeamBlerRow(
            estimator, detector, device, snr_db, seed, ratio, "all",
            0, 0, 0, 0, 0, 0, None, None, None, None, None, 0.0,
            snr_db, None, observation_basis, status, reason,
        )
    )
    return tuple(rows)

def _skipped_rows(
    *,
    estimator: str,
    detector: str,
    device: str,
    snr_db: float,
    seed: int,
    ratio: float,
    tx_settings: TxSettings,
    reference_snr_db: float,
    observation_basis: str,
    trigger_snr_db: float,
) -> tuple[BeamBlerRow, ...]:
    reason = (
        f"stop_at_zero_bler: higher SNR skipped after zero aggregate errors at "
        f"{trigger_snr_db:g} dB"
    )
    return tuple(
        BeamBlerRow(
            estimator, detector, device, snr_db, seed, ratio, user_name,
            0, 0, 0, 0, 0, 0, None, None, None, None, None, 0.0,
            reference_snr_db, None, observation_basis, "skipped", reason,
        )
        for user_name in (*(user.name for user in tx_settings.users), "all")
    )


def _noise_context(
    reference_power: float,
    array_gain: float,
    snr_db: float,
    ratio: float,
    weights: torch.Tensor,
) -> tuple[float, float, torch.Tensor, torch.Tensor | None]:
    array_variance = _array_noise_variance(reference_power, array_gain, snr_db, ratio)
    post_variance = ratio * array_variance
    covariance = beam_noise_covariance(weights, array_variance, post_variance)
    if not torch.isfinite(covariance).all().item():
        raise ValueError("R_eta 包含非有限值")
    hermitian_error = torch.max(torch.abs(covariance - covariance.conj().T)).item()
    if hermitian_error > 1e-6:
        raise ValueError("R_eta 不是 Hermitian 矩阵")
    eigenvalues = torch.linalg.eigvalsh(covariance).real
    if float(eigenvalues.min().item()) < -1e-6 * max(1.0, float(eigenvalues.abs().max().item())):
        raise ValueError("R_eta 不是半正定矩阵")
    chol, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
    weights_singular = False
    if ratio == 0.0:
        singular_values = torch.linalg.svdvals(weights)
        weights_singular = bool(
            (singular_values[-1] <= 1e-7 * singular_values[0]).item()
        )
    if weights_singular or int(info.max().item()) != 0:
        if ratio > 0.0:
            raise ValueError("post-combiner noise is positive but R_eta is not positive definite")
        return array_variance, post_variance, covariance, None
    if not torch.isfinite(chol).all().item():
        raise ValueError("R_eta Cholesky 因子包含非有限值")
    return array_variance, post_variance, covariance, chol


def _array_noise_variance(
    reference_power: float,
    array_gain: float,
    snr_db: float,
    ratio: float,
) -> float:
    try:
        gamma = math.pow(10.0, snr_db / 10.0)
    except OverflowError as exc:
        raise ValueError("snr_db 超出可计算范围") from exc
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("snr_db 超出可计算范围")
    variance = reference_power / (gamma * (array_gain + ratio))
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("SNR 标定产生非正或非有限阵元噪声方差")
    return variance


def _data_subcarrier_indices(transmitter: NrPuschTx) -> torch.Tensor:
    resource_grid = transmitter._tx_freq.resource_grid
    active = torch.zeros(resource_grid.fft_size, dtype=torch.bool, device=transmitter.device)
    effective = torch.as_tensor(
        resource_grid.effective_subcarrier_ind, dtype=torch.long, device=transmitter.device
    )
    active[effective] = True
    pilot_mask = transmitter._tx_freq.pilot_pattern.mask.bool()
    has_data = (~pilot_mask).any(dim=2).any(dim=1).any(dim=0)
    indices = torch.where(active & has_data)[0]
    if indices.numel() == 0:
        raise ValueError("TX resource grid 没有有效 PUSCH data RE")
    return indices


def _zf_rank_failure(
    channel: torch.Tensor,
    data_bins: torch.Tensor,
) -> str | None:
    matrices = channel.index_select(-1, data_bins).permute(2, 0, 1)
    singular_values = torch.linalg.svdvals(matrices)
    largest = singular_values[:, 0]
    ratios = torch.where(
        largest > 0.0,
        singular_values[:, -1] / largest,
        torch.zeros_like(largest),
    )
    failed = torch.where(ratios < 1e-7)[0]
    if failed.numel() == 0:
        return None
    local_index = int(failed[torch.argmin(ratios[failed])].item())
    fft_bin = int(data_bins[local_index].item())
    return (
        f"F 在 fft_bin={fft_bin} 的 sigma_min/sigma_max="
        f"{float(ratios[local_index].item()):.6g} < 1e-7"
    )


def _actual_snr_db(
    channel: torch.Tensor,
    covariance: torch.Tensor,
    data_bins: torch.Tensor,
) -> list[float | None]:
    diagonal_noise = covariance.diagonal().real
    result: list[float | None] = []
    for user in range(4):
        desired_power = channel[user, user].index_select(0, data_bins).abs().square().mean()
        noise_power = diagonal_noise[user]
        if (
            not torch.isfinite(desired_power).item()
            or not torch.isfinite(noise_power).item()
            or desired_power <= 0.0
            or noise_power <= 0.0
        ):
            result.append(None)
        else:
            result.append(float((10.0 * torch.log10(desired_power / noise_power)).item()))
    return result


def _diagonal_scale(
    tensor: torch.Tensor,
    covariance: torch.Tensor,
    *,
    beam_axis: int,
) -> torch.Tensor:
    diagonal = covariance.diagonal().real
    if (
        not torch.isfinite(diagonal).all().item()
        or torch.any(diagonal <= 0.0).item()
    ):
        raise ValueError("独立波束归一化要求每路 R_eta 对角元素为正有限值")
    shape = [1] * tensor.ndim
    shape[beam_axis] = 4
    return tensor / diagonal.sqrt().reshape(shape)


def _whiten(
    tensor: torch.Tensor,
    cholesky: torch.Tensor,
    *,
    beam_axis: int,
) -> torch.Tensor:
    moved = tensor.movedim(beam_axis, -1)
    if moved.shape[-1] != 4:
        raise ValueError("接收beam轴必须长度为4")
    rhs = moved.reshape(-1, 4).transpose(0, 1).contiguous()
    solved = torch.linalg.solve_triangular(cholesky, rhs, upper=False)
    return solved.transpose(0, 1).reshape(moved.shape).movedim(-1, beam_axis)


def _clopper_pearson(errors: int, frames: int, *, confidence: float) -> tuple[float, float]:
    if frames < 1 or not 0 <= errors <= frames or not 0.0 < confidence < 1.0:
        raise ValueError("Clopper-Pearson 区间参数无效")
    alpha = 1.0 - confidence
    lower = 0.0 if errors == 0 else float(
        scipy.stats.beta.ppf(alpha / 2.0, errors, frames - errors + 1)
    )
    upper = 1.0 if errors == frames else float(
        scipy.stats.beta.ppf(1.0 - alpha / 2.0, errors + 1, frames - errors)
    )
    return lower, upper




def _validate_ratio(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("post_combiner_ratio 必须为 [0,0.01] 内的有限数")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 0.01:
        raise ValueError("post_combiner_ratio 必须为 [0,0.01] 内的有限数")
    return result


def _validate_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed 必须为整数")
    return value


def _payload_seed(seed: int, frame_offset: int) -> int:
    return (seed + frame_offset) % (2**63 - 1)


def _noise_seed(seed: int, frame_offset: int) -> int:
    return (seed + 0x5EED5EED + frame_offset) % (2**63 - 1)


def _snapshot_hash(snapshot: RtBeamSnapshot) -> str:
    digest = hashlib.sha256()
    for key in ("h_ant", "h_beam", "taps_ant", "taps_beam", "weights", "path_a", "path_tau_s"):
        array = np.ascontiguousarray(getattr(snapshot, key))
        digest.update(key.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()
