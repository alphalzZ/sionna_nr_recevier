"""Run a configured SNR versus BLER sweep for four-user PUSCH."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from nr_pusch.bler import save_bler_results, simulate_detector_comparison
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.simulation_config import BlerSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate four-user NR PUSCH SNR versus BLER")
    parser.add_argument("--tx-config", required=True, help="TOML PUSCH transmit configuration")
    parser.add_argument("--channel-config", required=True, help="TOML CDL channel configuration")
    parser.add_argument("--simulation-config", required=True, help="TOML BLER simulation settings")
    parser.add_argument("--output", required=True, help="Output CSV path (JSON manifest is also written)")
    parser.add_argument("--device", default=None, help="Override configured device: cpu, cuda, cuda:0, or auto")
    parser.add_argument("--progress-jsonl", default=None, help="Append one JSON object per completed SNR point")
    args = parser.parse_args()

    tx_settings = TxSettings.from_toml(args.tx_config)
    channel_settings = ChannelSettings.from_toml(args.channel_config)
    simulation_settings = BlerSettings.from_toml(args.simulation_config)
    progress_path = Path(args.progress_jsonl) if args.progress_jsonl else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    def report_point(point) -> None:
        if progress_path is not None:
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(point), ensure_ascii=False) + "\n")

    points = simulate_detector_comparison(
        tx_settings, channel_settings, simulation_settings,
        device=args.device, on_point=report_point,
    )
    csv_path, manifest_path = save_bler_results(
        points,
        args.output,
        tx_settings=tx_settings,
        channel_settings=channel_settings,
        simulation_settings=simulation_settings,
    )
    print("Detector  Device  SNR [dB]  BLER      CRC fail  BER       TB errors / TBs  Runtime [s]")
    for point in points:
        print(
            f"{point.detector:9s} {point.device:6s} {point.snr_db:8.2f}  {point.bler:8.4g}  "
            f"{point.crc_fail_rate:8.4g}  {point.ber:8.4g}  "
            f"{point.block_errors}/{point.transport_blocks}  {point.runtime_s:.2f}"
        )
    print(f"CSV: {csv_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
