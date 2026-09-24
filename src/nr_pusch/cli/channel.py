"""Apply a configured CDL channel to an archived transmitter IQ result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nr_pusch.channel import NrPuschCdlChannel
from nr_pusch.channel_config import ChannelSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a 3GPP TR 38.901 CDL channel")
    parser.add_argument("--config", required=True, help="TOML CDL channel configuration")
    parser.add_argument("--input", required=True, help="Input NPZ containing tx IQ array 'iq'")
    parser.add_argument("--sample-rate-hz", required=True, type=int, help="Input IQ sample rate")
    parser.add_argument("--output", required=True, help="Output NPZ path")
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

    settings = ChannelSettings.from_toml(args.config)
    result = NrPuschCdlChannel(settings, device=args.device).apply(iq, args.sample_rate_hz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        iq=result.iq.detach().cpu().numpy().astype(np.complex64, copy=False),
        per_user_iq=result.per_user_iq.detach().cpu().numpy().astype(np.complex64, copy=False),
        channel_taps=result.channel_taps.detach().cpu().numpy().astype(np.complex64, copy=False),
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": input_path.name,
                "settings": settings.to_dict(),
                "result": result.metadata,
                "artifacts": {
                    "archive": output_path.name,
                    "arrays": ["iq", "per_user_iq", "channel_taps"],
                    "complex_dtype": "complex64",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Channelized IQ: {output_path}")
    print(f"IQ shape [batch,rx_antenna,sample]: {tuple(result.iq.shape)}")
    print(f"Per-user IQ shape [batch,user,rx_antenna,sample]: {tuple(result.per_user_iq.shape)}")


if __name__ == "__main__":
    main()
