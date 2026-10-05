"""Shared-array beam steering and channel-combination primitives."""

from __future__ import annotations

import math

import numpy as np


_NUM_BEAMS = 4


def make_beam_weights(
    element_positions_m: np.ndarray,
    steering_directions: np.ndarray,
    carrier_frequency_hz: float,
    *,
    phase_bits: int = 0,
) -> np.ndarray:
    """Return unit-norm receive weights for four world-coordinate directions.

    ``element_positions_m`` is ``[element, xyz]`` relative to the array phase
    center. ``steering_directions`` is ``[beam, xyz]`` and points from the
    phase center toward the desired source. Sionna's local y-z element order is
    preserved by the caller.
    """
    positions = np.asarray(element_positions_m)
    directions = np.asarray(steering_directions)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < _NUM_BEAMS:
        raise ValueError("element_positions_m 必须为 [element>=4,3]")
    if directions.shape != (_NUM_BEAMS, 3):
        raise ValueError("steering_directions 必须为 [4,3]")
    if not np.isrealobj(positions) or not np.isrealobj(directions):
        raise ValueError("阵列坐标和 steering direction 必须为实数")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(directions)):
        raise ValueError("阵列坐标和 steering direction 必须为有限值")
    if isinstance(carrier_frequency_hz, bool) or not math.isfinite(carrier_frequency_hz) or carrier_frequency_hz <= 0:
        raise ValueError("carrier_frequency_hz 必须为正有限值")
    if isinstance(phase_bits, bool) or not isinstance(phase_bits, int) or not 0 <= phase_bits <= 16:
        raise ValueError("phase_bits 必须为 0..16 的整数")

    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("steering direction 不能为零向量")
    unit_directions = directions / norms[:, None]
    wavelength_m = 299_792_458.0 / carrier_frequency_hz
    phase = 2.0 * np.pi * (positions @ unit_directions.T) / wavelength_m
    if phase_bits:
        phase_step = 2.0 * np.pi / (1 << phase_bits)
        phase = np.remainder(np.rint(phase / phase_step) * phase_step, 2.0 * np.pi)
    return np.exp(1j * phase) / math.sqrt(positions.shape[0])


def combine_beams(h_ant: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute ``Wᴴ H_ant`` with axes ``[beam, user, ...]``."""
    channel = np.asarray(h_ant)
    beam_weights = np.asarray(weights)
    if not np.iscomplexobj(channel) or not np.iscomplexobj(beam_weights):
        raise ValueError("h_ant 和 weights 必须为复数数组")
    if channel.ndim < 2:
        raise ValueError("h_ant 必须至少有 [element,user] 两个轴")
    if beam_weights.ndim != 2 or beam_weights.shape[0] != channel.shape[0] or beam_weights.shape[1] != _NUM_BEAMS:
        raise ValueError("weights 必须具有 [element,4] 轴且与 h_ant 阵元数一致")
    if not np.all(np.isfinite(channel)) or not np.all(np.isfinite(beam_weights)):
        raise ValueError("h_ant 和 weights 必须为有限值")
    return np.einsum("nb,nj...->bj...", beam_weights.conj(), channel, optimize=True)


def user_power_scales(power_scale_db: np.ndarray) -> np.ndarray:
    """Convert per-user relative dB settings to linear symbol-power scales."""
    values = np.asarray(power_scale_db)
    if values.ndim != 1 or not np.isrealobj(values) or not np.all(np.isfinite(values)):
        raise ValueError("power_scale_db 必须是一维有限实数数组")
    if np.any((values < -60.0) | (values > 60.0)):
        raise ValueError("power_scale_db 必须在 [-60,60] dB")
    return np.power(10.0, values / 10.0)


def compute_beam_metrics(
    h_beam: np.ndarray,
    power_scale_db: np.ndarray,
) -> dict[str, object]:
    """Summarize unwhitened beam leakage, power, correlation, and matrix rank."""
    channel = np.asarray(h_beam)
    if channel.ndim < 2 or channel.shape[:2] != (_NUM_BEAMS, _NUM_BEAMS):
        raise ValueError("h_beam 必须为 [4 beams,4 users,...]")
    if not np.iscomplexobj(channel) or not np.all(np.isfinite(channel)):
        raise ValueError("h_beam 必须为有限复数数组")
    power_db = np.asarray(power_scale_db)
    if power_db.shape != (_NUM_BEAMS,) or not np.isrealobj(power_db) or not np.all(np.isfinite(power_db)):
        raise ValueError("power_scale_db 必须为四个有限实数")
    powers = user_power_scales(power_db)
    channel_power = np.mean(np.abs(channel) ** 2, axis=tuple(range(2, channel.ndim)))

    leakage: list[list[float | None]] = []
    interference_to_signal: list[list[float | None]] = []
    interference_power: list[float] = []
    for beam in range(_NUM_BEAMS):
        leakage_row: list[float | None] = []
        isr_row: list[float | None] = []
        for user in range(_NUM_BEAMS):
            desired_user_power = channel_power[user, user]
            desired_beam_power = powers[beam] * channel_power[beam, beam]
            leakage_row.append(
                _ratio_or_none(channel_power[beam, user], desired_user_power)
            )
            isr_row.append(
                _ratio_or_none(powers[user] * channel_power[beam, user], desired_beam_power)
            )
        leakage.append(leakage_row)
        interference_to_signal.append(isr_row)
        interference_power.append(
            float(sum(powers[user] * channel_power[beam, user] for user in range(_NUM_BEAMS) if user != beam))
        )

    column_correlation: list[list[float | None]] = []
    for first in range(_NUM_BEAMS):
        row: list[float | None] = []
        first_column = channel[:, first, ...].reshape(-1)
        first_norm = float(np.linalg.norm(first_column))
        for second in range(_NUM_BEAMS):
            second_column = channel[:, second, ...].reshape(-1)
            denominator = first_norm * float(np.linalg.norm(second_column))
            row.append(
                None if denominator == 0.0 else float(abs(np.vdot(first_column, second_column)) / denominator)
            )
        column_correlation.append(row)

    matrices = channel.reshape(_NUM_BEAMS, _NUM_BEAMS, -1).transpose(2, 0, 1)
    matrices = matrices * np.sqrt(powers)[None, None, :]
    singular_values = np.linalg.svd(matrices, compute_uv=False)
    largest = singular_values[:, 0]
    tolerance = largest * np.finfo(np.float64).eps * _NUM_BEAMS
    ranks = np.sum(singular_values > tolerance[:, None], axis=1)
    ratios = np.divide(
        singular_values[:, -1],
        largest,
        out=np.zeros_like(largest),
        where=largest > 0.0,
    )
    return {
        "unpowered_mean_channel_power": channel_power.tolist(),
        "leakage_ratio_l_by_beam_user": leakage,
        "interference_to_signal_ratio_by_beam_user": interference_to_signal,
        "interference_power_by_beam": interference_power,
        "user_column_correlation": column_correlation,
        "matrix_sample_count": int(matrices.shape[0]),
        "rank_by_sample": ranks.astype(int).tolist(),
        "minimum_rank": int(ranks.min()),
        "maximum_rank": int(ranks.max()),
        "singular_value_ratio_by_sample": ratios.tolist(),
        "minimum_singular_value_ratio": float(ratios.min()),
        "condition_number_by_sample": [
            None if ratio == 0.0 else float(1.0 / ratio) for ratio in ratios
        ],
        "power_scale_db": power_db.astype(float).tolist(),
    }


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else float(numerator / denominator)
