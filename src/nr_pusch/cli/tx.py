"""Command-line PUSCH transmitter."""

from __future__ import annotations

import argparse

from nr_pusch.artifacts import save_tx_result
from nr_pusch.config import TxSettings
from nr_pusch.transmitter import NrPuschTx


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 4-UE NR PUSCH time-domain IQ")
    parser.add_argument("--config", required=True, help="TOML transmit configuration")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Sionna device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    settings = TxSettings.from_toml(args.config)
    transmitter = NrPuschTx(settings, device=args.device)
    result = transmitter.generate(args.batch_size, seed=args.seed)
    iq_path, manifest_path = save_tx_result(result, settings, args.output)
    print(f"IQ: {iq_path}")
    print(f"Manifest: {manifest_path}")
    print(f"IQ shape [batch,user,tx_antenna,sample]: {tuple(result.iq.shape)}")
    print(f"Transport block size per user: {transmitter.transport_block_size} bits")
    print(f"Sample rate: {result.sample_rate_hz} Hz")


if __name__ == "__main__":
    main()

