"""Run a configured SNR versus BLER sweep for multi-user PUSCH."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from nr_pusch.bler import BlerSweep, SkippedPoint, save_bler_results, simulate_detector_comparison
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.dmrs_prior import default_prior_dir
from nr_pusch.simulation_config import BlerSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate NR PUSCH SNR versus BLER")
    parser.add_argument("--tx-config", required=True, help="TOML PUSCH transmit configuration")
    parser.add_argument("--channel-config", required=True, help="TOML CDL channel configuration")
    parser.add_argument("--simulation-config", required=True, help="TOML BLER simulation settings")
    parser.add_argument("--output", required=True, help="Output CSV path (JSON manifest is also written)")
    parser.add_argument("--device", default=None, help="Override configured device: cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--progress-jsonl", default=None,
        help="Append one JSON object per completed or skipped SNR point",
    )
    parser.add_argument(
        "--prior-dir", default=None,
        help="共享 DMRS prior 目录，默认使用 --channel-config 同级的 tap_power_prior/",
    )
    args = parser.parse_args()

    tx_settings = TxSettings.from_toml(args.tx_config)
    channel_settings = ChannelSettings.from_toml(args.channel_config)
    channel_settings.validate_transmitter(tx_settings)
    simulation_settings = BlerSettings.from_toml(args.simulation_config)
    progress_path = Path(args.progress_jsonl) if args.progress_jsonl else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    def write_progress(record) -> None:
        if progress_path is not None:
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def report_point(point) -> None:
        write_progress({**asdict(point), "skipped": False})

    def report_skip(point: SkippedPoint) -> None:
        write_progress({**asdict(point), "skipped": True})

    sweep: BlerSweep = simulate_detector_comparison(
        tx_settings, channel_settings, simulation_settings,
        prior_dir=(
            Path(args.prior_dir) if args.prior_dir is not None
            else default_prior_dir(args.channel_config)
        ),
        device=args.device, on_point=report_point, on_skip=report_skip,
    )
    csv_path, manifest_path = save_bler_results(
        sweep,
        args.output,
        tx_settings=tx_settings,
        channel_settings=channel_settings,
        simulation_settings=simulation_settings,
    )
    print(
        f"Topology: {len(tx_settings.users)} users × {tx_settings.pusch.num_layers} "
        f"layers/user = {len(tx_settings.users) * tx_settings.pusch.num_layers} streams; "
        f"{tx_settings.pusch.num_antenna_ports} Tx/user → "
        f"{channel_settings.antennas.rx_num_rows * channel_settings.antennas.rx_num_cols} Rx"
    )
    print("Estimator       Detector        Device  SNR [dB]  BLER      CRC fail  BER       TB errors / TBs  Runtime [s]")
    for point in sweep.points:
        print(
            f"{point.channel_estimator:13s} {point.detector:14s} {point.device:6s} "
            f"{point.snr_db:8.2f}  {point.bler:8.4g}  {point.crc_fail_rate:8.4g}  "
            f"{point.ber:8.4g}  {point.block_errors}/{point.transport_blocks}  {point.runtime_s:.2f}"
        )
    if sweep.skipped:
        print("Skipped SNR points (BLER reached 0):")
        for point in sweep.skipped:
            print(
                f"{point.channel_estimator:13s} {point.detector:14s} {point.device:6s} "
                f"{point.snr_db:8.2f}  skipped after BLER 0 at {point.trigger_snr_db:.2f} dB "
                f"({point.reason})"
            )
    print(f"CSV: {csv_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
