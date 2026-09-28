"""Decode a captured or simulated four-antenna PUSCH IQ archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nr_pusch.config import TxSettings
from nr_pusch.iq import read_matlab_rx_reference
from nr_pusch.receiver import NrPuschRx


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and decode four-user DFT-s-OFDM PUSCH captures")
    parser.add_argument(
        "--rx-config", "--tx-config", dest="rx_config", required=True,
        help="TOML PUSCH profile used to configure the receiver",
    )
    parser.add_argument(
        "--input", required=True, help="NPZ capture or MATLAB HDF5 receive test vector"
    )
    parser.add_argument(
        "--input-format", choices=("auto", "npz", "matlab-h5"), default="auto",
        help="Input file format; auto detects HDF5 by extension",
    )
    parser.add_argument("--output", required=True, help="Output NPZ with decoded bits and CRC status")
    parser.add_argument("--noise-variance", required=True, type=float, help="AWGN variance per complex sample")
    parser.add_argument("--channel-estimator", choices=("dmrs", "perfect"), default="dmrs")
    parser.add_argument("--input-domain", choices=("time", "frequency"), default=None)
    parser.add_argument(
        "--detector", choices=("lmmse", "lmmse-sic", "k-best", "ep", "mmse-pic"), default="lmmse"
    )
    parser.add_argument("--detector-parameter", type=int, default=None)
    parser.add_argument("--detector-damping", type=float, default=0.25)
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
    input_format = args.input_format
    if input_format == "auto":
        input_format = "matlab-h5" if input_path.suffix.lower() in {".h5", ".hdf5"} else "npz"
    input_domain = args.input_domain or ("frequency" if input_format == "matlab-h5" else "time")
    if input_format == "matlab-h5" and input_domain != "frequency":
        parser.error("MATLAB H5 接收网格是频域数据，--input-domain 必须为 frequency")
    if input_format == "matlab-h5" and args.channel_estimator == "perfect":
        parser.error("MATLAB H5 夹具不包含真实信道；请使用 --channel-estimator dmrs")
    if output_path.suffix.lower() != ".npz":
        parser.error("--output 必须使用 .npz 后缀")
    channel = None
    input_analysis: dict[str, object]
    if input_format == "matlab-h5":
        try:
            reference = read_matlab_rx_reference(input_path)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"无法读取 MATLAB H5 接收数据: {exc}")
        grid = reference.frequency_grid
        received = torch.from_numpy(grid[None, None, ...])
        input_analysis = {
            "dataset": "FreqData/IQdataPdu_real + FreqData/IQdataPdu_imag",
            "axis_order": ["batch", "stream", "rx_antenna", "ofdm_symbol", "active_subcarrier"],
            "grid_shape": list(received.shape),
            "antenna_mean_power": np.mean(np.abs(grid) ** 2, axis=(1, 2)).tolist(),
            "peak_magnitude": float(np.abs(grid).max()),
            "data_grid_shape": list(reference.data_grid.shape) if reference.data_grid is not None else None,
            "pilot_grid_shape": list(reference.pilot_grid.shape) if reference.pilot_grid is not None else None,
        }
    else:
        try:
            with np.load(input_path, allow_pickle=False) as archive:
                if input_domain == "time":
                    if "iq" not in archive:
                        parser.error("时域输入 NPZ 缺少 'iq' 数组")
                    received = torch.from_numpy(np.array(archive["iq"], copy=True))
                    channel = (
                        torch.from_numpy(np.array(archive["channel_taps"], copy=True))
                        if "channel_taps" in archive else None
                    )
                else:
                    if "grid" not in archive:
                        parser.error("频域输入 NPZ 缺少 'grid' 数组")
                    received = torch.from_numpy(np.array(archive["grid"], copy=True))
                    channel = (
                        torch.from_numpy(np.array(archive["channel_frequency_response"], copy=True))
                        if "channel_frequency_response" in archive else None
                    )
                input_analysis = {
                    "array": "iq" if input_domain == "time" else "grid",
                    "shape": list(received.shape),
                    "dtype": str(received.dtype),
                }
        except (OSError, ValueError) as exc:
            parser.error(f"无法读取 NPZ 接收数据: {exc}")

    settings = TxSettings.from_toml(args.rx_config)
    receiver = NrPuschRx(
        settings,
        channel_estimator=args.channel_estimator,
        detector=args.detector,
        detector_parameter=args.detector_parameter,
        detector_damping=args.detector_damping,
        input_domain=input_domain,
        max_delay_spread_s=args.max_delay_spread_s,
        device=args.device,
    )
    if input_domain == "time":
        result = receiver.receive(received, args.noise_variance, channel_taps=channel)
    else:
        result = receiver.receive_frequency_grid(
            received,
            args.noise_variance,
            channel_frequency_response=channel,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        bits=result.bits.detach().cpu().numpy().astype(np.uint8, copy=False),
        crc_status=result.crc_status.detach().cpu().numpy().astype(np.bool_, copy=False),
        constellation_real=result.constellation.detach().cpu().numpy().real.astype(np.float32, copy=False),
        constellation_imag=result.constellation.detach().cpu().numpy().imag.astype(np.float32, copy=False),
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": input_path.name,
                "input_format": input_format,
                "input_domain": input_domain,
                "input_analysis": input_analysis,
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
    for index, passed in enumerate(crc.reshape(-1).tolist()):
        print(f"UE {index}: {'PASS' if passed else 'FAIL'}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
