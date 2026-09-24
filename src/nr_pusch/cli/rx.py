"""Decode a captured or simulated four-antenna PUSCH IQ archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nr_pusch.config import TxSettings
from nr_pusch.receiver import NrPuschRx


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode four-user DFT-s-OFDM PUSCH IQ")
    parser.add_argument("--tx-config", required=True, help="TOML PUSCH profile used for decoding")
    parser.add_argument("--input", required=True, help="NPZ containing complex receive IQ array 'iq'")
    parser.add_argument("--output", required=True, help="Output NPZ with decoded bits and CRC status")
    parser.add_argument("--noise-variance", required=True, type=float, help="AWGN variance per complex sample")
    parser.add_argument("--channel-estimator", choices=("dmrs", "perfect"), default="dmrs")
    parser.add_argument(
        "--max-delay-spread-s",
        type=float,
        default=3e-6,
        help="Maximum delay spread used to size the DFT-s DMRS channel basis",
    )
    parser.add_argument("--device", default=None, help="Sionna device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".npz":
        parser.error("--output 必须使用 .npz 后缀")
    with np.load(input_path) as archive:
        if "iq" not in archive:
            parser.error("输入 NPZ 缺少 'iq' 数组")
        iq = torch.from_numpy(np.array(archive["iq"], copy=True))
        channel_taps = (
            torch.from_numpy(np.array(archive["channel_taps"], copy=True))
            if "channel_taps" in archive
            else None
        )

    settings = TxSettings.from_toml(args.tx_config)
    receiver = NrPuschRx(
        settings,
        channel_estimator=args.channel_estimator,
        max_delay_spread_s=args.max_delay_spread_s,
        device=args.device,
    )
    result = receiver.receive(iq, args.noise_variance, channel_taps=channel_taps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        bits=result.bits.detach().cpu().numpy().astype(np.uint8, copy=False),
        crc_status=result.crc_status.detach().cpu().numpy().astype(np.bool_, copy=False),
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": input_path.name,
                "settings": settings.to_dict(),
                "receiver": result.metadata,
                "noise_variance": args.noise_variance,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    crc = result.crc_status.detach().cpu()
    print(f"Decoded blocks: {crc.numel()}")
    print(f"CRC pass: {int(crc.sum())}/{crc.numel()}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
