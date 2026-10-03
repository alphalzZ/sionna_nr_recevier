"""Monte Carlo simulation of four-user PUSCH block error rate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import time
from dataclasses import replace
from typing import Any, Callable

import sionna.phy
import torch
from sionna.phy.channel import time_lag_discrete_time_channel

from .channel import NrPuschCdlChannel
from .channel_config import ChannelSettings
from .config import TxSettings
from .device import use_device
from .dmrs_prior import (
    dmrs_prior_compatibility,
    load_dmrs_tap_power_prior,
)
from .noise import add_awgn, add_awgn_resource_grid
from .receiver import NrPuschRx, _build_dmrs_frequency_basis
from .simulation_config import BlerSettings
from .transmitter import NrPuschTx


@dataclass(frozen=True)
class BlerPoint:
    detector: str
    device: str
    snr_db: float
    frames: int
    transport_blocks: int
    block_errors: int
    crc_failures: int
    bit_errors: int
    bits: int
    bler: float
    crc_fail_rate: float
    ber: float
    runtime_s: float
    channel_estimator: str


@dataclass(frozen=True)
class SkippedPoint:
    """An SNR point that was not simulated because an earlier point reached BLER 0."""

    detector: str
    device: str
    snr_db: float
    trigger_snr_db: float
    reason: str
    channel_estimator: str


@dataclass(frozen=True)
class BlerSweep:
    """Measured SNR points together with the points skipped by the stop policy."""

    points: tuple[BlerPoint, ...]
    skipped: tuple[SkippedPoint, ...]

def estimate_dmrs_tap_power_prior(
    tx_settings: TxSettings,
    channel_settings: ChannelSettings,
    *,
    l_min: int,
    max_delay_spread_s: float,
    num_realizations: int,
    seed: int,
    device: str | None,
) -> torch.Tensor:
    """Estimate diagonal tap powers from independent CDL channel realizations."""
    if num_realizations < 1:
        raise ValueError("num_realizations 必须为正整数")
    if max_delay_spread_s <= 0:
        raise ValueError("max_delay_spread_s 必须大于 0")
    channel_settings.validate_transmitter(tx_settings)
    device = use_device(device)
    sionna.phy.config.seed = seed
    torch.manual_seed(seed)
    transmitter = NrPuschTx(tx_settings, device=device)
    resource_grid = transmitter._tx_freq.resource_grid
    sample_rate_hz = transmitter.sample_rate_hz
    _, l_max = time_lag_discrete_time_channel(sample_rate_hz, max_delay_spread_s)
    basis = _build_dmrs_frequency_basis(
        resource_grid.fft_size, l_min, int(l_max), device=torch.device(device)
    )
    channel = NrPuschCdlChannel(channel_settings, device=device)
    power_sum = torch.zeros(basis.shape[-1], dtype=torch.float64, device=device)
    count = 0
    batch_size = min(8, num_realizations)
    for start in range(0, num_realizations, batch_size):
        current_batch = min(batch_size, num_realizations - start)
        empty_grid = torch.zeros(
            (
                current_batch,
                len(tx_settings.users),
                tx_settings.pusch.num_antenna_ports,
                resource_grid.num_ofdm_symbols,
                resource_grid.fft_size,
            ),
            dtype=torch.complex64,
            device=device,
        )
        response = channel.apply_frequency(empty_grid, resource_grid).channel_frequency_response
        # The configured validation channel is static over a slot. Use one
        # response per UE/RX/realization to avoid counting OFDM symbols as
        # additional independent channel draws.
        response = response[:, 0, :, :, :, 0, :]
        response_matrix = response.reshape(-1, resource_grid.fft_size).T.contiguous()
        taps = torch.linalg.lstsq(basis, response_matrix).solution
        power_sum += taps.abs().square().to(torch.float64).sum(dim=1)
        count += taps.shape[1]
    tap_power = (power_sum / count).to(torch.float32)
    floor = tap_power.max() * 1e-8
    tap_power = tap_power.clamp_min(floor)
    if not torch.isfinite(tap_power).all().item() or not torch.all(tap_power > 0).item():
        raise RuntimeError("CDL tap-power calibration produced an invalid prior")
    return tap_power


def simulate_bler(
    tx_settings: TxSettings,
    channel_settings: ChannelSettings,
    simulation_settings: BlerSettings,
    *,
    device: str | None = None,
    on_point: Callable[[BlerPoint], None] | None = None,
    on_skip: Callable[[SkippedPoint], None] | None = None,
) -> BlerSweep:
    """Run the configured SNR sweep; perfect CDL CSI is the default baseline.

    With ``stop_at_zero_bler`` enabled, a detector stops sweeping once a point
    measures no block error, and the remaining higher SNRs are reported as
    skipped instead of simulated. This assumes BLER does not grow with SNR.
    """
    simulation_settings.validate()
    channel_settings.validate_transmitter(tx_settings)
    device = use_device(device or simulation_settings.device)
    sionna.phy.config.seed = simulation_settings.seed
    torch.manual_seed(simulation_settings.seed)

    transmitter = NrPuschTx(tx_settings, device=device)
    channel = NrPuschCdlChannel(channel_settings, device=device)
    max_delay_spread_s = (
        simulation_settings.max_delay_spread_s
        if simulation_settings.max_delay_spread_s is not None
        else channel_settings.channel.max_delay_spread_s
    )
    dmrs_tap_power_prior = None
    if simulation_settings.channel_estimator == "dmrs-lmmse":
        if simulation_settings.dmrs_tap_power_prior_path is None:
            raise ValueError("dmrs-lmmse 配置必须设置 dmrs_tap_power_prior_path")
        compatibility = dmrs_prior_compatibility(
            tx_settings,
            channel_settings,
            l_min=simulation_settings.l_min,
            max_delay_spread_s=max_delay_spread_s,
            fft_size=transmitter._tx_freq.resource_grid.fft_size,
            sample_rate_hz=transmitter.sample_rate_hz,
        )
        dmrs_tap_power_prior, _ = load_dmrs_tap_power_prior(
            simulation_settings.dmrs_tap_power_prior_path,
            expected_compatibility=compatibility,
            device=device,
        )
    rx = NrPuschRx(
        tx_settings,
        channel_estimator=simulation_settings.channel_estimator,
        dmrs_tap_power_prior=dmrs_tap_power_prior,
        l_min=simulation_settings.l_min,
        max_delay_spread_s=max_delay_spread_s,
        num_decoder_iterations=simulation_settings.num_decoder_iterations,
        detector=simulation_settings.detector,
        detector_parameter=simulation_settings.detector_parameters.get(
            simulation_settings.detector, simulation_settings.detector_parameter
        ),
        detector_damping=simulation_settings.detector_damping,
        estimate_delay=simulation_settings.estimate_delay,
        input_domain=("frequency" if simulation_settings.channel_domain == "frequency" else "time"),
        device=device,
    )
    if transmitter.sample_rate_hz != rx.sample_rate_hz:
        raise RuntimeError("发送端与接收端 sample rate 不一致")

    points: list[BlerPoint] = []
    skipped: list[SkippedPoint] = []
    effective_batch_size = simulation_settings.batch_size_for_detector()
    frame_seed = simulation_settings.seed
    for snr_index, snr_db in enumerate(simulation_settings.snr_db):
        started = time.perf_counter()
        frames = transport_blocks = block_errors = crc_failures = bit_errors = bits = 0
        while (
            frames < simulation_settings.max_frames_per_snr
            and block_errors < simulation_settings.target_block_errors
        ):
            batch_size = min(
                effective_batch_size,
                simulation_settings.max_frames_per_snr - frames,
            )
            tx_result = transmitter.generate(batch_size=batch_size, seed=frame_seed)
            frame_seed += batch_size
            if simulation_settings.channel_domain == "frequency":
                channel_result = channel.apply_frequency(
                    tx_result.frequency_grid,
                    transmitter._tx_freq.resource_grid,
                )
                noisy = add_awgn_resource_grid(
                    channel_result.grid,
                    snr_db,
                    seed=simulation_settings.seed + frame_seed,
                )
                rx_result = rx.receive_frequency_grid(
                    noisy.grid,
                    noisy.noise_variance,
                    channel_frequency_response=(
                        channel_result.channel_frequency_response
                        if simulation_settings.channel_estimator == "perfect"
                        else None
                    ),
                )
            else:
                channel_result = channel.apply(tx_result.iq, tx_result.sample_rate_hz)
                noisy = add_awgn(
                    channel_result.iq,
                    snr_db,
                    seed=simulation_settings.seed + frame_seed,
                )
                rx_result = rx.receive(
                    noisy.iq,
                    noisy.noise_variance,
                    channel_taps=(
                        channel_result.channel_taps
                        if simulation_settings.channel_estimator == "perfect"
                        else None
                    ),
                )
            crc = rx_result.crc_status
            payload_errors = torch.any(rx_result.bits != tx_result.bits, dim=-1)
            frame_errors = torch.logical_or(~crc, payload_errors)
            transport_blocks += int(crc.numel())
            block_errors += int(frame_errors.sum().item())
            crc_failures += int((~crc).sum().item())
            bit_errors += int((rx_result.bits != tx_result.bits).sum().item())
            bits += int(tx_result.bits.numel())
            frames += batch_size

        points.append(
            BlerPoint(
                detector=simulation_settings.detector,
                device=device,
                snr_db=float(snr_db),
                frames=frames,
                transport_blocks=transport_blocks,
                block_errors=block_errors,
                crc_failures=crc_failures,
                bit_errors=bit_errors,
                bits=bits,
                bler=block_errors / transport_blocks,
                crc_fail_rate=crc_failures / transport_blocks,
                ber=bit_errors / bits,
                runtime_s=time.perf_counter() - started,
                channel_estimator=simulation_settings.channel_estimator,
            )
        )
        if on_point is not None:
            on_point(points[-1])
        if simulation_settings.stop_at_zero_bler and block_errors == 0:
            for remaining_snr_db in simulation_settings.snr_db[snr_index + 1:]:
                skipped_point = SkippedPoint(
                    detector=simulation_settings.detector,
                    device=device,
                    snr_db=float(remaining_snr_db),
                    trigger_snr_db=float(snr_db),
                    reason="bler_at_zero",
                    channel_estimator=simulation_settings.channel_estimator,
                )
                skipped.append(skipped_point)
                if on_skip is not None:
                    on_skip(skipped_point)
            break
    return BlerSweep(points=tuple(points), skipped=tuple(skipped))


def simulate_detector_comparison(
    tx_settings: TxSettings,
    channel_settings: ChannelSettings,
    simulation_settings: BlerSettings,
    *,
    device: str | None = None,
    on_point: Callable[[BlerPoint], None] | None = None,
    on_skip: Callable[[SkippedPoint], None] | None = None,
) -> BlerSweep:
    """Run each configured estimator-detector pair from the same seed and profile."""
    points: list[BlerPoint] = []
    skipped: list[SkippedPoint] = []
    for channel_estimator in simulation_settings.channel_estimators_for_sweep:
        for detector in simulation_settings.detectors:
            run_settings = replace(
                simulation_settings,
                channel_estimator=channel_estimator,
                channel_estimators=(channel_estimator,),
                dmrs_tap_power_prior_path=(
                    simulation_settings.dmrs_tap_power_prior_path
                    if channel_estimator == "dmrs-lmmse"
                    else None
                ),
                detector=detector,
                detectors=(detector,),
            )
            sweep = simulate_bler(
                tx_settings, channel_settings, run_settings,
                device=device, on_point=on_point, on_skip=on_skip,
            )
            points.extend(sweep.points)
            skipped.extend(sweep.skipped)
    return BlerSweep(points=tuple(points), skipped=tuple(skipped))


def save_bler_results(
    sweep: BlerSweep,
    output: str | Path,
    *,
    tx_settings: TxSettings,
    channel_settings: ChannelSettings,
    simulation_settings: BlerSettings,
) -> tuple[Path, Path]:
    output = Path(output)
    if output.suffix.lower() != ".csv":
        raise ValueError("BLER 曲线结果文件必须使用 .csv 后缀")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(BlerPoint.__dataclass_fields__)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(point) for point in sweep.points)
    manifest_path = output.with_suffix(".json")
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "transmit_settings": tx_settings.to_dict(),
        "channel_settings": channel_settings.to_dict(),
        "simulation_settings": simulation_settings.to_dict(),
        "results": [asdict(point) for point in sweep.points],
        "skipped_points": [asdict(point) for point in sweep.skipped],
        "skip_policy": (
            "stop_at_zero_bler=true ends each estimator/detector pair at its first SNR point with "
            "zero block errors; the remaining higher SNRs are listed in skipped_points and are not simulated"
        ),
        "bler_definition": "TB block error if CRC fails or any decoded payload bit differs",
        "channel_domain": simulation_settings.channel_domain,
        "detector_validity_note": (
            "CP-OFDM uses native Sionna per-resource-element detection; K-best is per RE, EP uses "
            "double precision, and MMSE-PIC feeds extrinsic LLRs between per-RE iterations. "
            "No DFT despreading is applied."
            if tx_settings.pusch.waveform == "cp_ofdm"
            else
            "DFT-s-OFDM k-best first applies frequency-domain LMMSE, then uses a per-sample zero-lag "
            "effective spatial channel with residual frequency variation in the covariance. DFT-s-OFDM "
            "MMSE-PIC uses soft time-domain moments for iterative frequency-domain cancellation without "
            "LDPC feedback; soft-mmse-pic feeds rate-matched LDPC extrinsic LLRs back into soft PIC. "
            "The DFT-s-OFDM EP detector applies damped Gaussian-site moment matching to the same per-sample "
            "spatial model; residual frequency variation is approximated as Gaussian covariance."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output, manifest_path
