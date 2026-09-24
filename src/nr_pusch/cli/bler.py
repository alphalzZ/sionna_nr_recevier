"""Run a configured SNR versus BLER sweep for four-user PUSCH."""

from __future__ import annotations

import argparse

from nr_pusch.bler import save_bler_results, simulate_bler
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.simulation_config import BlerSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate four-user NR PUSCH SNR versus BLER")
    parser.add_argument("--tx-config", required=True, help="TOML PUSCH transmit configuration")
    parser.add_argument("--channel-config", required=True, help="TOML CDL channel configuration")
    parser.add_argument("--simulation-config", required=True, help="TOML BLER simulation settings")
    parser.add_argument("--output", required=True, help="Output CSV path (JSON manifest is also written)")
    parser.add_argument("--device", default=None, help="Sionna device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    tx_settings = TxSettings.from_toml(args.tx_config)
    channel_settings = ChannelSettings.from_toml(args.channel_config)
    simulation_settings = BlerSettings.from_toml(args.simulation_config)
    points = simulate_bler(tx_settings, channel_settings, simulation_settings, device=args.device)
    csv_path, manifest_path = save_bler_results(
        points,
        args.output,
        tx_settings=tx_settings,
        channel_settings=channel_settings,
        simulation_settings=simulation_settings,
    )
    print("SNR [dB]  BLER      CRC fail  BER       TB errors / TBs")
    for point in points:
        print(
            f"{point.snr_db:8.2f}  {point.bler:8.4g}  {point.crc_fail_rate:8.4g}  "
            f"{point.ber:8.4g}  {point.block_errors}/{point.transport_blocks}"
        )
    print(f"CSV: {csv_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
