"""Monte Carlo simulation of four-user PUSCH block error rate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import time
from dataclasses import replace
from typing import Any

import sionna.phy
import torch

from .channel import NrPuschCdlChannel
from .channel_config import ChannelSettings
from .config import TxSettings
from .noise import add_awgn
from .receiver import NrPuschRx
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


def simulate_bler(
    tx_settings: TxSettings,
    channel_settings: ChannelSettings,
    simulation_settings: BlerSettings,
    *,
    device: str | None = None,
) -> list[BlerPoint]:
    """Run the configured SNR sweep; perfect CDL CSI is the default baseline."""
    simulation_settings.validate()
    device = _resolve_device(device or simulation_settings.device)
    sionna.phy.config.seed = simulation_settings.seed
    torch.manual_seed(simulation_settings.seed)

    transmitter = NrPuschTx(tx_settings, device=device)
    channel = NrPuschCdlChannel(channel_settings, device=device)
    rx = NrPuschRx(
        tx_settings,
        channel_estimator=simulation_settings.channel_estimator,
        max_delay_spread_s=channel_settings.channel.max_delay_spread_s,
        num_decoder_iterations=simulation_settings.num_decoder_iterations,
        detector=simulation_settings.detector,
        detector_parameter=simulation_settings.detector_parameter,
        device=device,
    )
    if transmitter.sample_rate_hz != rx.sample_rate_hz:
        raise RuntimeError("发送端与接收端 sample rate 不一致")

    points: list[BlerPoint] = []
    tb_size = transmitter.transport_block_size
    frame_seed = simulation_settings.seed
    for snr_db in simulation_settings.snr_db:
        started = time.perf_counter()
        frames = transport_blocks = block_errors = crc_failures = bit_errors = bits = 0
        while (
            frames < simulation_settings.max_frames_per_snr
            and block_errors < simulation_settings.target_block_errors
        ):
            batch_size = min(
                simulation_settings.batch_size,
                simulation_settings.max_frames_per_snr - frames,
            )
            tx_result = transmitter.generate(batch_size=batch_size, seed=frame_seed)
            frame_seed += batch_size
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
            )
        )
    return points


def simulate_detector_comparison(
    tx_settings: TxSettings,
    channel_settings: ChannelSettings,
    simulation_settings: BlerSettings,
    *,
    device: str | None = None,
) -> list[BlerPoint]:
    """Run each configured detector from the same seed and channel profile."""
    points: list[BlerPoint] = []
    for detector in simulation_settings.detectors:
        run_settings = replace(simulation_settings, detector=detector, detectors=(detector,))
        points.extend(simulate_bler(tx_settings, channel_settings, run_settings, device=device))
    return points


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("配置请求 CUDA，但当前 PyTorch 环境没有可用 GPU/CUDA")
    return requested


def save_bler_results(
    points: list[BlerPoint],
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
        writer.writerows(asdict(point) for point in points)
    manifest_path = output.with_suffix(".json")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "transmit_settings": tx_settings.to_dict(),
        "channel_settings": channel_settings.to_dict(),
        "simulation_settings": simulation_settings.to_dict(),
        "results": [asdict(point) for point in points],
        "bler_definition": "TB block error if CRC fails or any decoded payload bit differs",
        "detector_validity_note": (
            "For DFT-s-OFDM, per-RE QAM assumptions in k-best, EP, and MMSE-PIC are experimental; "
            "their BLER must not be treated as validated until cross-subcarrier detection is implemented."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output, manifest_path
