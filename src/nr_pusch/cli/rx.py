"""Decode a captured or simulated four-antenna PUSCH IQ archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import numpy as np
import torch

from nr_pusch.config import TxSettings
from nr_pusch.iq import read_matlab_rx_reference, read_matlab_scrambling_sequences
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
    parser.add_argument("--noise-variance", type=float, default=None, help="AWGN variance; defaults to [receiver] in RX TOML")
    parser.add_argument("--channel-estimator", choices=("dmrs", "perfect"), default=None)
    parser.add_argument("--input-domain", choices=("time", "frequency"), default=None)
    parser.add_argument(
        "--detector", choices=("lmmse", "lmmse-sic", "k-best", "ep", "mmse-pic"), default=None
    )
    parser.add_argument("--detector-parameter", type=int, default=None)
    parser.add_argument("--detector-damping", type=float, default=None)
    parser.add_argument(
        "--max-delay-spread-s",
        type=float,
        default=None,
        help="Maximum delay spread used to size the DFT-s DMRS channel basis",
    )
    parser.add_argument("--device", default=None, help="Sionna device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--scrambling", default=None,
        help="MATLAB HDF5 file with per-UE scrambling sequences (ue<k>_scrambSeq) for non-RNTI c_init",
    )
    parser.add_argument(
        "--spatial-denoise", action="store_true", default=None,
        help="Apply dominant-spatial-signature projection to the DMRS channel estimate",
    )
    parser.add_argument(
        "--estimate-delay", action="store_true", default=None,
        help="Estimate and report the per-antenna bulk delay of the DMRS estimate",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    try:
        with Path(args.rx_config).open("rb") as config_file:
            config_profile = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        parser.error(f"无法读取 RX TOML 配置: {exc}")
    receiver_profile = config_profile.get("receiver", {})
    if not isinstance(receiver_profile, dict):
        parser.error("RX TOML 的 [receiver] 必须是表")
    noise_variance = args.noise_variance
    if noise_variance is None:
        noise_variance = float(receiver_profile.get("noise_variance", 0.0))
    channel_estimator = args.channel_estimator or receiver_profile.get("channel_estimator", "dmrs")
    detector = args.detector or receiver_profile.get("detector", "lmmse")
    detector_parameter = (
        args.detector_parameter
        if args.detector_parameter is not None
        else receiver_profile.get("detector_parameter")
    )
    detector_damping = (
        args.detector_damping
        if args.detector_damping is not None
        else float(receiver_profile.get("detector_damping", 0.25))
    )
    max_delay_spread_s = (
        args.max_delay_spread_s
        if args.max_delay_spread_s is not None
        else float(receiver_profile.get("max_delay_spread_s", 3e-6))
    )
    input_format = args.input_format
    if input_format == "auto":
        input_format = "matlab-h5" if input_path.suffix.lower() in {".h5", ".hdf5"} else "npz"
    input_domain = args.input_domain or receiver_profile.get(
        "input_domain", "frequency" if input_format == "matlab-h5" else "time"
    )
    if input_format == "matlab-h5" and input_domain != "frequency":
        parser.error("MATLAB H5 接收网格是频域数据，--input-domain 必须为 frequency")
    if input_format == "matlab-h5" and channel_estimator == "perfect":
        parser.error("MATLAB H5 夹具不包含真实信道；请使用 --channel-estimator dmrs")
    if output_path.suffix.lower() != ".npz":
        parser.error("--output 必须使用 .npz 后缀")
    channel = None
    transmitted_bits = None
    input_analysis: dict[str, object]
    if input_format == "matlab-h5":
        try:
            reference = read_matlab_rx_reference(input_path)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"无法读取 MATLAB H5 接收数据: {exc}")
        grid = reference.frequency_grid
        transmitted_bits = reference.transmitted_bits
        received = torch.from_numpy(grid[None, None, ...])
        input_analysis = {
            "dataset": "FreqData/IQdataPdu_real + FreqData/IQdataPdu_imag",
            "axis_order": ["batch", "stream", "rx_antenna", "ofdm_symbol", "active_subcarrier"],
            "grid_shape": list(received.shape),
            "antenna_mean_power": np.mean(np.abs(grid) ** 2, axis=(1, 2)).tolist(),
            "peak_magnitude": float(np.abs(grid).max()),
            "data_grid_shape": list(reference.data_grid.shape) if reference.data_grid is not None else None,
            "pilot_grid_shape": list(reference.pilot_grid.shape) if reference.pilot_grid is not None else None,
            "transmitted_bits_shape": list(transmitted_bits.shape) if transmitted_bits is not None else None,
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

    spatial_denoise = args.spatial_denoise
    if spatial_denoise is None:
        spatial_denoise = bool(receiver_profile.get("spatial_denoise", False))
    estimate_delay = args.estimate_delay
    if estimate_delay is None:
        estimate_delay = bool(receiver_profile.get("estimate_delay", False))
    scrambling_sequences = None
    if args.scrambling is not None:
        try:
            sequences = read_matlab_scrambling_sequences(args.scrambling)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"无法读取加扰序列: {exc}")
        scrambling_sequences = torch.from_numpy(sequences)
    settings = TxSettings.from_toml(args.rx_config)
    receiver = NrPuschRx(
        settings,
        channel_estimator=channel_estimator,
        detector=detector,
        detector_parameter=detector_parameter,
        detector_damping=detector_damping,
        input_domain=input_domain,
        max_delay_spread_s=max_delay_spread_s,
        device=args.device,
        scrambling_sequences=scrambling_sequences,
        spatial_denoise=spatial_denoise,
        estimate_delay=estimate_delay,
    )
    if input_domain == "time":
        result = receiver.receive(received, noise_variance, channel_taps=channel)
    else:
        result = receiver.receive_frequency_grid(
            received,
            noise_variance,
            channel_frequency_response=channel,
        )
    reference_comparison = None
    if transmitted_bits is not None:
        decoded_bits = result.bits.detach().cpu().numpy().astype(np.uint8, copy=False)
        if decoded_bits.shape[0] != 1 or decoded_bits.shape[1:] != transmitted_bits.shape:
            parser.error(
                f"译码 bits 形状 {decoded_bits.shape} 与 H5 参考形状 {transmitted_bits.shape} 不匹配"
            )
        bit_errors = np.count_nonzero(decoded_bits[0] != transmitted_bits, axis=-1)
        crc_status = result.crc_status.detach().cpu().numpy().astype(bool).reshape(-1)
        reference_comparison = {
            "bit_errors": bit_errors.tolist(),
            "ber": (bit_errors / transmitted_bits.shape[-1]).tolist(),
            "bit_count_per_user": int(transmitted_bits.shape[-1]),
            "exact_match": bool(np.all(bit_errors == 0)),
            "crc_status": crc_status.tolist(),
            # The H5 payload comes from the reference link, so a user whose own
            # CRC passes here can still differ when that link failed to decode
            # the same transport block. CRC status is the trustworthy verdict.
            "note": (
                "bit_errors compare against the reference link payload; a user with crc_status true "
                "but non-zero bit_errors indicates the reference link failed on that block, not that "
                "this decode is wrong"
            ),
            "crc_verified_users": int(crc_status.sum()),
        }
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
                "scrambling_source": (
                    str(args.scrambling) if args.scrambling is not None else "rnti"
                ),
                "input_analysis": input_analysis,
                "settings": settings.to_dict(),
                "receiver": result.metadata,
                "noise_variance": noise_variance,
                "reference_comparison": reference_comparison,
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
    if reference_comparison is not None:
        print(f"Reference bit errors: {reference_comparison['bit_errors']}")
        mismatched = [
            index
            for index, errors in enumerate(reference_comparison["bit_errors"])
            if errors and reference_comparison["crc_status"][index]
        ]
        if mismatched:
            print(
                "Note: users "
                + ", ".join(str(index) for index in mismatched)
                + " pass CRC here but differ from the reference payload;"
                " the reference link likely failed on those blocks"
            )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
