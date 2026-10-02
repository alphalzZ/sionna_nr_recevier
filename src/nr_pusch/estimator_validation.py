"""Paired three-arm validation of DMRS channel estimators."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import sionna.phy
import torch

from .bler import estimate_dmrs_tap_power_prior
from .channel import NrPuschCdlChannel
from .channel_config import ChannelSettings
from .config import TxSettings
from .device import use_device
from .dmrs_prior import dmrs_prior_compatibility, save_dmrs_tap_power_prior
from .noise import add_awgn_resource_grid
from .receiver import NrPuschRx
from .simulation_config import BlerSettings
from .transmitter import NrPuschTx, TxResult


_ESTIMATORS = ("dmrs", "dmrs-lmmse", "perfect")
_SPLITS = ("development", "holdout", "high-snr")


def run_estimator_validation(
    tx_settings: TxSettings,
    channel_settings: ChannelSettings,
    simulation_settings: BlerSettings,
    validation_settings: dict[str, Any],
    *,
    output: str | Path,
    device: str | None = None,
    prior_realizations: int | None = None,
    frames_per_snr: int | None = None,
    on_progress: Callable[[str, float, int, float], None] | None = None,
) -> dict[str, Any]:
    """Train a separate CDL prior and compare baseline/candidate/perfect CSI."""
    simulation_settings.validate()
    if simulation_settings.channel_estimator != "dmrs":
        raise ValueError("估计器验证配置的 [bler].channel_estimator 必须固定为 dmrs")
    if simulation_settings.detector != "soft-mmse-pic":
        raise ValueError("估计器验证必须统一使用 soft-mmse-pic 检测器")
    if simulation_settings.detectors != ("soft-mmse-pic",):
        raise ValueError("估计器验证只接受单一 soft-mmse-pic 检测器臂")
    if (
        simulation_settings.detector_parameters.get(
            "soft-mmse-pic", simulation_settings.detector_parameter
        )
        != 1
        or simulation_settings.detector_damping != 0.25
        or simulation_settings.num_decoder_iterations != 20
    ):
        raise ValueError("估计器验证固定 soft-mmse-pic=1、阻尼=0.25、译码迭代=20")
    if simulation_settings.channel_domain != "frequency":
        raise ValueError("估计器验证只支持已统一的频域信道链路")
    if tuple(simulation_settings.snr_db) != (25.0, 30.0):
        raise ValueError("估计器验证主 SNR 必须严格为 25 和 30 dB")
    training_seed = _integer_setting(validation_settings, "training_seed")
    development_seed = _integer_setting(validation_settings, "development_seed")
    holdout_seed = _integer_setting(validation_settings, "holdout_seed")
    bootstrap_seed = _integer_setting(validation_settings, "bootstrap_seed")
    training_realizations = _positive_override(
        prior_realizations, validation_settings, "training_realizations", "prior-realizations"
    )
    development_frames = _positive_override(
        frames_per_snr, validation_settings, "development_frames_per_snr", "frames-per-snr"
    )
    holdout_frames = _positive_override(
        frames_per_snr, validation_settings, "holdout_frames_per_snr", "frames-per-snr"
    )
    high_snr_frames = _integer_setting(validation_settings, "high_snr_frames")
    bootstrap_replicates = _integer_setting(validation_settings, "bootstrap_replicates")
    high_snr_db = float(validation_settings["high_snr_db"])
    if not np.isfinite(high_snr_db) or high_snr_db <= max(simulation_settings.snr_db):
        raise ValueError("high_snr_db 必须高于主验证 SNR")
    if min(training_realizations, development_frames, holdout_frames, high_snr_frames) < 1:
        raise ValueError("training/development/holdout frame 数必须为正整数")
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates 至少为 100")

    device = use_device(device or simulation_settings.device)
    transmitter = NrPuschTx(tx_settings, device=device)
    resource_grid = transmitter._tx_freq.resource_grid
    sample_rate_hz = transmitter.sample_rate_hz
    max_delay_spread_s = (
        simulation_settings.max_delay_spread_s
        if simulation_settings.max_delay_spread_s is not None
        else channel_settings.channel.max_delay_spread_s
    )
    prior = estimate_dmrs_tap_power_prior(
        tx_settings,
        channel_settings,
        l_min=simulation_settings.l_min,
        max_delay_spread_s=max_delay_spread_s,
        num_realizations=training_realizations,
        seed=training_seed,
        device=device,
    )
    compatibility = dmrs_prior_compatibility(
        tx_settings,
        channel_settings,
        l_min=simulation_settings.l_min,
        max_delay_spread_s=max_delay_spread_s,
        fft_size=resource_grid.fft_size,
        sample_rate_hz=sample_rate_hz,
    )
    output_path = Path(output)
    if output_path.suffix.lower() != ".json":
        raise ValueError("--output 必须使用 .json 后缀")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path = output_path.with_name(output_path.stem + ".prior.npz")
    save_dmrs_tap_power_prior(
        prior_path,
        prior,
        compatibility=compatibility,
        training_seed=training_seed,
        training_realizations=training_realizations,
    )

    decoder_iterations = simulation_settings.num_decoder_iterations
    detector_parameter = simulation_settings.detector_parameters.get(
        "soft-mmse-pic", simulation_settings.detector_parameter
    )
    receivers = {
        name: NrPuschRx(
            tx_settings,
            channel_estimator=name,
            dmrs_tap_power_prior=prior if name == "dmrs-lmmse" else None,
            l_min=simulation_settings.l_min,
            max_delay_spread_s=max_delay_spread_s,
            num_decoder_iterations=decoder_iterations,
            detector="soft-mmse-pic",
            detector_parameter=detector_parameter,
            detector_damping=simulation_settings.detector_damping,
            input_domain="frequency",
            device=device,
        )
        for name in _ESTIMATORS
    }
    if any(receiver.sample_rate_hz != sample_rate_hz for receiver in receivers.values()):
        raise RuntimeError("发送端与接收端 sample rate 不一致")

    mask_tx = transmitter.generate(batch_size=1, seed=training_seed)
    pilot_mask = transmitter._tx_freq.pilot_pattern.mask.to(device=device).bool()
    active_re = mask_tx.frequency_grid[0].abs().any(dim=1)
    data_masks = active_re[:, None] & ~pilot_mask
    precoder = (
        torch.as_tensor(
            np.stack([cfg.precoding_matrix for cfg in tx_settings.to_sionna_configs()]),
            dtype=mask_tx.frequency_grid.dtype, device=device,
        )
        if tx_settings.pusch.precoding == "codebook" else None
    )
    if not torch.any(data_masks).item():
        raise RuntimeError("无法从配置的资源栅格确定 PUSCH data RE")

    snr_values = (*simulation_settings.snr_db, high_snr_db)
    frame_counts = (development_frames, holdout_frames, high_snr_frames)
    max_frames = max(frame_counts)
    num_users = len(tx_settings.users)
    num_rx_ant = channel_settings.antennas.rx_num_rows * channel_settings.antennas.rx_num_cols
    shape = (len(_ESTIMATORS), len(_SPLITS), len(snr_values), max_frames, num_users)
    block_error = np.zeros(shape, dtype=np.bool_)
    crc_failure = np.zeros(shape, dtype=np.bool_)
    bit_errors = np.zeros(shape, dtype=np.int32)
    data_re_nmse = np.full(shape + (num_rx_ant,), np.nan, dtype=np.float64)
    runtime_s = np.full(shape[:-1], np.nan, dtype=np.float64)
    frame_seed = np.full((len(_SPLITS), len(snr_values), max_frames), -1, dtype=np.int64)

    split_seeds = (development_seed, holdout_seed, holdout_seed + 1)
    split_started = time.perf_counter()
    for split_index, (split, split_seed, num_frames) in enumerate(
        zip(_SPLITS, split_seeds, frame_counts, strict=True)
    ):
        sionna.phy.config.seed = split_seed
        torch.manual_seed(split_seed)
        channel = NrPuschCdlChannel(channel_settings, device=device)
        snr_indices = (2,) if split == "high-snr" else (0, 1)
        for snr_index in snr_indices:
            snr_db = snr_values[snr_index]
            started = time.perf_counter()
            batch_size = simulation_settings.batch_size
            for frame_start in range(0, num_frames, batch_size):
                current_batch = min(batch_size, num_frames - frame_start)
                frame_stop = frame_start + current_batch
                payload_seeds = [split_seed + index for index in range(frame_start, frame_stop)]
                frame_seed[split_index, snr_index, frame_start:frame_stop] = payload_seeds
                frame_results = [
                    transmitter.generate(batch_size=1, seed=payload_seed)
                    for payload_seed in payload_seeds
                ]
                tx_result = TxResult(
                    iq=torch.cat([result.iq for result in frame_results], dim=0),
                    frequency_grid=torch.cat(
                        [result.frequency_grid for result in frame_results], dim=0
                    ),
                    bits=torch.cat([result.bits for result in frame_results], dim=0),
                    sample_rate_hz=frame_results[0].sample_rate_hz,
                    metadata=frame_results[0].metadata,
                )
                channel_result = channel.apply_frequency(tx_result.frequency_grid, resource_grid)
                noisy = add_awgn_resource_grid(
                    channel_result.grid,
                    snr_db,
                    seed=split_seed + (snr_index + 1) * 1_000_003 + frame_start,
                )
                truth = channel_result.channel_frequency_response
                effective_truth = (
                    torch.einsum("bxruasf,ual->bxrulsf", truth, precoder)
                    if precoder is not None else truth
                )
                for estimator_index, estimator in enumerate(_ESTIMATORS):
                    receiver = receivers[estimator]
                    started_frame = time.perf_counter()
                    rx_result = receiver.receive_frequency_grid(
                        noisy.grid,
                        noisy.noise_variance,
                        channel_frequency_response=truth if estimator == "perfect" else None,
                    )
                    estimated_h = (
                        effective_truth
                        if estimator == "perfect"
                        else receiver._estimator.last_channel_estimate
                    )
                    if estimated_h is None:
                        raise RuntimeError(f"{estimator} estimator did not retain its CSI estimate")
                    nmse = _data_re_nmse(estimated_h, effective_truth, data_masks)
                    crc = rx_result.crc_status.detach().cpu().numpy().reshape(
                        current_batch, num_users
                    )
                    errors = (
                        (rx_result.bits != tx_result.bits)
                        .sum(dim=-1)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(current_batch, num_users)
                    )
                    crc_failure[
                        estimator_index, split_index, snr_index, frame_start:frame_stop
                    ] = ~crc
                    bit_errors[
                        estimator_index, split_index, snr_index, frame_start:frame_stop
                    ] = errors
                    block_error[
                        estimator_index, split_index, snr_index, frame_start:frame_stop
                    ] = (~crc) | (errors > 0)
                    data_re_nmse[
                        estimator_index, split_index, snr_index, frame_start:frame_stop
                    ] = nmse
                    runtime_s[
                        estimator_index, split_index, snr_index, frame_start:frame_stop
                    ] = (time.perf_counter() - started_frame) / current_batch
            elapsed = time.perf_counter() - started
            if on_progress is not None:
                on_progress(split, float(snr_db), num_frames, elapsed)

    frame_path = output_path.with_name(output_path.stem + ".frames.npz")
    np.savez_compressed(
        frame_path,
        frame_seed=frame_seed,
        block_error=block_error,
        crc_failure=crc_failure,
        bit_errors=bit_errors,
        data_re_nmse=data_re_nmse,
        runtime_s=runtime_s,
        estimators=np.asarray(_ESTIMATORS),
        splits=np.asarray(_SPLITS),
        snr_db=np.asarray(snr_values, dtype=np.float64),
        frame_counts=np.asarray(frame_counts, dtype=np.int32),
    )
    metrics = _summarize(
        block_error,
        crc_failure,
        bit_errors,
        data_re_nmse,
        runtime_s,
        frame_counts,
        num_users,
        transmitter.transport_block_size,
        [user.name for user in tx_settings.users],
    )
    acceptance = _acceptance_gate(
        block_error,
        data_re_nmse,
        frame_counts,
        bootstrap_replicates,
        bootstrap_seed,
    )
    summary = {
        "estimators": list(_ESTIMATORS),
        "splits": list(_SPLITS),
        "snr_db": list(snr_values),
        "frame_counts": dict(zip(_SPLITS, frame_counts, strict=True)),
        "split_seeds": dict(zip(_SPLITS, split_seeds, strict=True)),
        "training": {
            "seed": training_seed,
            "realizations": training_realizations,
            "tap_power": prior.detach().cpu().tolist(),
            "prior_path": str(prior_path),
        },
        "configuration": {
            "tx": tx_settings.to_dict(),
            "channel": channel_settings.to_dict(),
            "bler": simulation_settings.to_dict(),
            "compatibility": compatibility,
            "detector": "soft-mmse-pic",
            "detector_parameter": detector_parameter,
            "detector_damping": simulation_settings.detector_damping,
        },
        "metrics": metrics,
        "acceptance_gate": acceptance,
        "adoption": (
            "dmrs-lmmse remains explicit opt-in; no existing default changed"
            if acceptance["passed"]
            else "not adopted; existing dmrs default and profiles remain unchanged"
        ),
        "artifacts": {
            "summary": str(output_path),
            "frames": str(frame_path),
            "prior": str(prior_path),
        },
        "runtime_s": time.perf_counter() - split_started,
    }
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def _positive_override(
    override: int | None, settings: dict[str, Any], key: str, flag: str
) -> int:
    if override is None:
        return _integer_setting(settings, key)
    if isinstance(override, bool) or not isinstance(override, int) or override < 1:
        raise ValueError(f"--{flag} 必须为正整数")
    return override


def _integer_setting(settings: dict[str, Any], key: str) -> int:
    value = settings.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"estimator_validation.{key} 必须为正整数")
    return value


def _data_re_nmse(
    estimated: torch.Tensor,
    truth: torch.Tensor,
    data_masks: torch.Tensor,
) -> np.ndarray:
    result = np.empty(
        (estimated.shape[0], data_masks.shape[0], estimated.shape[2]),
        dtype=np.float64,
    )
    for batch_index in range(estimated.shape[0]):
        for user in range(data_masks.shape[0]):
            for rx_ant in range(estimated.shape[2]):
                mask = data_masks[user]
                estimate_user = estimated[batch_index, 0, rx_ant, user][mask]
                truth_user = truth[batch_index, 0, rx_ant, user][mask]
                power = truth_user.abs().square().sum()
                result[batch_index, user, rx_ant] = (
                    float((estimate_user - truth_user).abs().square().sum().item()
                          / power.item())
                    if power.item() > 0 else float("nan")
                )
    return result


def _summarize(
    block_error: np.ndarray,
    crc_failure: np.ndarray,
    bit_errors: np.ndarray,
    data_re_nmse: np.ndarray,
    runtime_s: np.ndarray,
    frame_counts: tuple[int, ...],
    num_users: int,
    transport_block_size: int,
    user_names: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for estimator_index, estimator in enumerate(_ESTIMATORS):
        summary[estimator] = {}
        for split_index, split in enumerate(_SPLITS):
            summary[estimator][split] = {}
            snr_indices = (2,) if split == "high-snr" else (0, 1)
            for snr_index in snr_indices:
                count = frame_counts[split_index]
                errors = block_error[estimator_index, split_index, snr_index, :count]
                crc = crc_failure[estimator_index, split_index, snr_index, :count]
                bits = bit_errors[estimator_index, split_index, snr_index, :count]
                nmse = data_re_nmse[estimator_index, split_index, snr_index, :count]
                per_user = []
                for user_index, user_name in enumerate(user_names):
                    user_block_errors = errors[:, user_index]
                    user_crc_failures = crc[:, user_index]
                    user_bit_errors = bits[:, user_index]
                    per_user.append(
                        {
                            "user": user_name,
                            "block_errors": int(user_block_errors.sum()),
                            "bler": float(user_block_errors.mean()),
                            "crc_failures": int(user_crc_failures.sum()),
                            "crc_failure_rate": float(user_crc_failures.mean()),
                            "bit_errors": int(user_bit_errors.sum()),
                            "ber": float(
                                user_bit_errors.sum() / (count * transport_block_size)
                            ),
                            "data_re_nmse_mean_by_rx": np.nanmean(
                                nmse[:, user_index, :], axis=0
                            ).tolist(),
                        }
                    )
                summary[estimator][split][str((25.0, 30.0, 60.0)[snr_index])] = {
                    "frames": count,
                    "transport_blocks": count * num_users,
                    "block_errors": int(errors.sum()),
                    "bler": float(errors.sum() / (count * num_users)),
                    "crc_failures": int(crc.sum()),
                    "crc_failure_rate": float(crc.sum() / (count * num_users)),
                    "bit_errors": int(bits.sum()),
                    "ber": float(bits.sum() / (count * num_users * transport_block_size)),
                    "per_user": per_user,
                    "data_re_nmse_mean_by_user_rx": np.nanmean(nmse, axis=0).tolist(),
                    "runtime_s": float(
                        np.nansum(runtime_s[estimator_index, split_index, snr_index, :count])
                    ),
                }
    for split_index, split in enumerate(_SPLITS):
        snr_indices = (2,) if split == "high-snr" else (0, 1)
        for snr_index in snr_indices:
            snr_key = str((25.0, 30.0, 60.0)[snr_index])
            perfect_bler = summary["perfect"][split][snr_key]["bler"]
            for estimator in _ESTIMATORS[:-1]:
                summary[estimator][split][snr_key]["perfect_csi_bler_gap"] = (
                    summary[estimator][split][snr_key]["bler"] - perfect_bler
                )
    return summary


def _acceptance_gate(
    block_error: np.ndarray,
    data_re_nmse: np.ndarray,
    frame_counts: tuple[int, ...],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    holdout_index = _SPLITS.index("holdout")
    checks: dict[str, Any] = {}
    passes = True
    for snr_index in (0, 1):
        count = frame_counts[holdout_index]
        baseline = block_error[0, holdout_index, snr_index, :count]
        candidate = block_error[1, holdout_index, snr_index, :count]
        baseline_errors = int(baseline.sum())
        candidate_errors = int(candidate.sum())
        relative_reduction = (
            1.0 - candidate_errors / baseline_errors if baseline_errors > 0 else None
        )
        frame_difference = candidate.sum(axis=1).astype(np.float64) - baseline.sum(axis=1)
        upper = _paired_bootstrap_upper(
            frame_difference,
            bootstrap_replicates,
            bootstrap_seed + snr_index,
            num_users=baseline.shape[1],
        )
        baseline_nmse = np.nanmean(data_re_nmse[0, holdout_index, snr_index, :count])
        candidate_nmse = np.nanmean(data_re_nmse[1, holdout_index, snr_index, :count])
        check = {
            "baseline_block_errors": baseline_errors,
            "candidate_block_errors": candidate_errors,
            "relative_bler_reduction": relative_reduction,
            "paired_bler_difference_97_5pct_upper": upper,
            "baseline_data_re_nmse": float(baseline_nmse),
            "candidate_data_re_nmse": float(candidate_nmse),
            "passed": bool(
                relative_reduction is not None
                and relative_reduction >= 0.10
                and upper < 0.0
                and candidate_nmse < baseline_nmse
            ),
        }
        checks[str((25.0, 30.0)[snr_index])] = check
        passes &= check["passed"]
    high_index = _SPLITS.index("high-snr")
    high_count = frame_counts[high_index]
    baseline_high_errors = int(block_error[0, high_index, 2, :high_count].sum())
    candidate_high_errors = int(block_error[1, high_index, 2, :high_count].sum())
    high_passed = candidate_high_errors <= baseline_high_errors
    checks["60.0"] = {
        "baseline_block_errors": baseline_high_errors,
        "candidate_block_errors": candidate_high_errors,
        "passed": bool(high_passed),
    }
    return {"checks": checks, "passed": bool(passes and high_passed)}


def _paired_bootstrap_upper(
    frame_difference: np.ndarray,
    replicates: int,
    seed: int,
    *,
    num_users: int,
) -> float:
    rng = np.random.default_rng(seed)
    count = frame_difference.size
    samples = np.empty(replicates, dtype=np.float64)
    offset = 0
    while offset < replicates:
        batch = min(128, replicates - offset)
        indices = rng.integers(0, count, size=(batch, count))
        samples[offset : offset + batch] = frame_difference[indices].mean(axis=1) / num_users
        offset += batch
    return float(np.quantile(samples, 0.975))
