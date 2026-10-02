"""Apply a configured CDL channel to archived time or frequency-domain TX data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nr_pusch.channel import NrPuschCdlChannel
from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.transmitter import NrPuschTx


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a 3GPP TR 38.901 CDL channel")
    parser.add_argument("--config", required=True, help="TOML CDL channel configuration")
    parser.add_argument("--input", required=True, help="Input NPZ containing tx 'iq' or 'frequency_grid'")
    parser.add_argument("--sample-rate-hz", required=True, type=int, help="Input IQ sample rate")
    parser.add_argument("--domain", choices=("time", "frequency"), default="time")
    parser.add_argument("--tx-config", help="TX profile for topology checks; required in frequency mode")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--device", default=None, help="Sionna device, e.g. cpu or cuda:0")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".npz":
        parser.error("--output 必须使用 .npz 后缀")
    settings = ChannelSettings.from_toml(args.config)
    tx_settings = TxSettings.from_toml(args.tx_config) if args.tx_config else None
    if tx_settings is not None:
        settings.validate_transmitter(tx_settings)
    channel = NrPuschCdlChannel(settings, device=args.device)
    if args.domain == "time":
        with np.load(input_path) as archive:
            if "iq" not in archive:
                parser.error("时域模式输入 NPZ 缺少 'iq' 数组")
            iq = torch.from_numpy(np.array(archive["iq"], copy=True))
        if tx_settings is not None:
            if iq.shape[1] != len(tx_settings.users):
                parser.error("iq 用户数与 --tx-config 不一致")
            if args.sample_rate_hz != NrPuschTx(tx_settings, device=args.device).sample_rate_hz:
                parser.error("--sample-rate-hz 与 --tx-config 的采样率不一致")
        result = channel.apply(iq, args.sample_rate_hz)
        arrays = {
            "iq": result.iq.detach().cpu().numpy().astype(np.complex64, copy=False),
            "per_user_iq": result.per_user_iq.detach().cpu().numpy().astype(np.complex64, copy=False),
            "channel_taps": result.channel_taps.detach().cpu().numpy().astype(np.complex64, copy=False),
        }
        axis_info = {
            "iq": ["batch", "rx_antenna", "sample"],
            "per_user_iq": ["batch", "user", "rx_antenna", "sample"],
            "channel_taps": result.metadata["channel_tap_axes"],
        }
        shape_summary = f"IQ shape [batch,rx_antenna,sample]: {tuple(result.iq.shape)}"
    else:
        if args.tx_config is None:
            parser.error("frequency 模式必须提供 --tx-config")
        with np.load(input_path) as archive:
            if "frequency_grid" not in archive:
                parser.error("频域模式输入 NPZ 缺少 'frequency_grid' 数组")
            frequency_grid = torch.from_numpy(np.array(archive["frequency_grid"], copy=True))
        tx = NrPuschTx(tx_settings, device=args.device)
        if frequency_grid.shape[1] != len(tx_settings.users):
            parser.error("frequency_grid 用户数与 --tx-config 不一致")
        if args.sample_rate_hz != tx.sample_rate_hz:
            parser.error("--sample-rate-hz 与 --tx-config 的资源栅格采样率不一致")
        result = channel.apply_frequency(frequency_grid, tx._tx_freq.resource_grid)
        arrays = {
            "grid": result.grid.detach().cpu().numpy().astype(np.complex64, copy=False),
            "per_user_grid": result.per_user_grid.detach().cpu().numpy().astype(np.complex64, copy=False),
            "channel_frequency_response": result.channel_frequency_response.detach().cpu().numpy().astype(
                np.complex64, copy=False
            ),
        }
        axis_info = {
            "grid": result.metadata["grid_axes"],
            "per_user_grid": ["batch", "user", "rx_antenna", "ofdm_symbol", "fft_bin"],
            "channel_frequency_response": result.metadata["channel_axes"],
        }
        shape_summary = f"Grid shape [batch,num_rx,rx_antenna,symbol,fft]: {tuple(result.grid.shape)}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": input_path.name,
                "settings": settings.to_dict(),
                "result": result.metadata,
                "artifacts": {
                    "archive": output_path.name,
                    "domain": args.domain,
                    "arrays": axis_info,
                    "complex_dtype": "complex64",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Channelized {args.domain} output: {output_path}")
    print(shape_summary)
    print(
        f"Topology: {result.metadata['num_users']} users × "
        f"{result.metadata.get('num_tx_antennas_per_user', result.metadata.get('tx_antennas_per_user'))} "
        f"Tx antennas → {result.metadata['num_rx_antennas']} Rx antennas"
    )


if __name__ == "__main__":
    main()
