"""Decode a captured or simulated PUSCH IQ archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import numpy as np
import torch

from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.dmrs_prior import (
    default_prior_dir,
    dmrs_prior_compatibility,
    resolve_dmrs_tap_power_prior,
)
from nr_pusch.iq import read_matlab_rx_reference, read_matlab_scrambling_sequences
from nr_pusch.receiver import NrPuschRx
from nr_pusch.transmitter import NrPuschTx


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and decode configurable NR PUSCH captures")
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
    parser.add_argument(
        "--input-domain", choices=("time", "frequency"), default=None,
        help="NPZ sample domain; defaults to the RX TOML profile",
    )
    parser.add_argument("--output", required=True, help="Output NPZ with decoded bits and CRC status")
    parser.add_argument("--noise-variance", type=float, default=None, help="AWGN variance; defaults to [receiver] in RX TOML")
    parser.add_argument("--channel-estimator", choices=("dmrs", "dmrs-lmmse", "perfect"), default=None)
    parser.add_argument(
        "--channel-config",
        default=None,
        help="CDL TOML matching the tap-power prior; required for dmrs-lmmse",
    )
    parser.add_argument(
        "--prior-dir",
        default=None,
        help="共享 DMRS prior 目录，默认使用信道配置同级的 tap_power_prior/",
    )
    parser.add_argument(
        "--detector",
        choices=("lmmse", "lmmse-sic", "k-best", "ep", "mmse-pic", "soft-mmse-pic"),
        default=None,
    )
    parser.add_argument("--detector-parameter", type=int, default=None)
    parser.add_argument("--detector-damping", type=float, default=None)
    parser.add_argument(
        "--l-min", type=int, default=None,
        help="Minimum discrete-time DMRS channel tap (default -6)",
    )
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
        "--cb-crc", action="store_true", default=None,
        help="Record the per-code-block CRC verdict for every user",
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
    profile_detector = receiver_profile.get("detector", "lmmse")
    detector = args.detector or profile_detector
    detector_overridden = args.detector is not None and args.detector != profile_detector
    detector_parameter = (
        args.detector_parameter
        if args.detector_parameter is not None
        else None
        if detector_overridden and detector == "soft-mmse-pic"
        else receiver_profile.get("detector_parameter")
    )
    detector_damping = (
        args.detector_damping
        if args.detector_damping is not None
        else 0.25
        if detector_overridden and detector == "soft-mmse-pic"
        else float(receiver_profile.get("detector_damping", 0.25))
    )
    max_delay_spread_s = (
        args.max_delay_spread_s
        if args.max_delay_spread_s is not None
        else float(receiver_profile.get("max_delay_spread_s", 3e-6))
    )
    l_min = (
        args.l_min if args.l_min is not None
        else int(receiver_profile.get("l_min", -6))
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

    track_cb_crc = args.cb_crc
    if track_cb_crc is None:
        track_cb_crc = bool(receiver_profile.get("cb_crc", False))
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
    if input_format == "matlab-h5" and (
        len(settings.users) != 4 or settings.pusch.num_layers != 1
        or settings.pusch.num_antenna_ports != 1
    ):
        parser.error("MATLAB H5 夹具仅支持四用户、单层、单发射天线的固定捕获格式")
    if args.channel_config is not None and channel_estimator != "dmrs-lmmse":
        ChannelSettings.from_toml(args.channel_config).validate_transmitter(settings)
    if "dmrs_tap_power_prior_path" in receiver_profile:
        parser.error(
            "dmrs_tap_power_prior_path 已移除；先验改为按信道配置在共享先验目录中自动查找"
        )
    dmrs_tap_power_prior = None
    if channel_estimator == "dmrs-lmmse":
        channel_config_value = args.channel_config or receiver_profile.get(
            "dmrs_tap_power_prior_channel_config_path"
        )
        if not isinstance(channel_config_value, str) or not channel_config_value:
            parser.error("dmrs-lmmse 严格兼容性检查要求 --channel-config")
        channel_config_path = Path(channel_config_value)
        if (
            args.channel_config is None
            and not channel_config_path.is_absolute()
        ):
            channel_config_path = Path(args.rx_config).resolve().parent / channel_config_path
        prior_dir = (
            Path(args.prior_dir)
            if args.prior_dir is not None
            else default_prior_dir(channel_config_path)
        )
        try:
            channel_settings = ChannelSettings.from_toml(channel_config_path)
            channel_settings.validate_transmitter(settings)
            tx_preview = NrPuschTx(settings, device=args.device)
            compatibility = dmrs_prior_compatibility(
                settings,
                channel_settings,
                l_min=l_min,
                max_delay_spread_s=max_delay_spread_s,
                fft_size=tx_preview._tx_freq.resource_grid.fft_size,
                sample_rate_hz=tx_preview.sample_rate_hz,
            )
            dmrs_tap_power_prior, _, _ = resolve_dmrs_tap_power_prior(
                prior_dir,
                expected_compatibility=compatibility,
                device=args.device,
            )
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"无法加载匹配的 DMRS tap-power prior: {exc}")
    receiver = NrPuschRx(
        settings,
        channel_estimator=channel_estimator,
        dmrs_tap_power_prior=dmrs_tap_power_prior,
        detector=detector,
        detector_parameter=detector_parameter,
        detector_damping=detector_damping,
        input_domain=input_domain,
        max_delay_spread_s=max_delay_spread_s,
        l_min=l_min,
        device=args.device,
        scrambling_sequences=scrambling_sequences,
        estimate_delay=estimate_delay,
        track_cb_crc=track_cb_crc,
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
    print(
        f"Topology: {len(settings.users)} users × {settings.pusch.num_layers} "
        f"layers/user; {settings.pusch.num_antenna_ports} Tx/user → "
        f"{result.metadata['num_rx_antennas']} Rx antennas"
    )
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
    if result.metadata.get("cb_crc_status"):
        for user, status in enumerate(result.metadata["cb_crc_status"]):
            passed = sum(1 for value in status if value)
            failed = [str(index) for index, value in enumerate(status) if not value]
            suffix = f" (failed: {', '.join(failed)})" if failed else ""
            print(f"CB CRC ue{user}: {passed}/{len(status)} code blocks passed{suffix}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
