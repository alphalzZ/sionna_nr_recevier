"""Run paired validation of DMRS estimators with one shared CDL/AWGN sample."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings
from nr_pusch.dmrs_prior import default_prior_dir
from nr_pusch.estimator_validation import run_estimator_validation
from nr_pusch.simulation_config import BlerSettings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a CDL tap prior and run paired DMRS estimator validation"
    )
    parser.add_argument("--tx-config", required=True, help="TOML PUSCH transmit configuration")
    parser.add_argument("--channel-config", required=True, help="TOML CDL channel configuration")
    parser.add_argument(
        "--validation-config", required=True, help="TOML BLER and estimator-validation settings"
    )
    parser.add_argument("--output", required=True, help="JSON summary output path")
    parser.add_argument("--device", default=None, help="Override configured device")
    parser.add_argument(
        "--prior-dir", default=None,
        help="共享 DMRS prior 目录，默认使用 --channel-config 同级的 tap_power_prior/",
    )
    parser.add_argument(
        "--prior-realizations", type=int, default=None,
        help="Override CDL prior training realizations for smoke validation",
    )
    parser.add_argument(
        "--frames-per-snr", type=int, default=None,
        help="Override development and holdout frame counts for smoke validation",
    )
    args = parser.parse_args()
    tx_settings = TxSettings.from_toml(args.tx_config)
    prior_dir = (
        Path(args.prior_dir)
        if args.prior_dir is not None
        else default_prior_dir(args.channel_config)
    )
    channel_settings = ChannelSettings.from_toml(args.channel_config)
    simulation_settings = BlerSettings.from_toml(args.validation_config)
    with Path(args.validation_config).open("rb") as stream:
        validation_profile = tomllib.load(stream)
    validation_settings = validation_profile.get("estimator_validation")
    if not isinstance(validation_settings, dict):
        parser.error("validation TOML 必须包含 [estimator_validation] 表")

    def report(split: str, snr_db: float, frames: int, elapsed: float) -> None:
        print(
            f"{split:12s} {snr_db:5.1f} dB  {frames:5d} frames  {elapsed:.1f} s",
            flush=True,
        )

    summary = run_estimator_validation(
        tx_settings,
        channel_settings,
        simulation_settings,
        validation_settings,
        output=args.output,
        prior_dir=prior_dir,
        device=args.device,
        prior_realizations=args.prior_realizations,
        frames_per_snr=args.frames_per_snr,
        on_progress=report,
    )
    print(f"Acceptance gate: {'PASS' if summary['acceptance_gate']['passed'] else 'FAIL'}")
    print(f"Summary: {summary['artifacts']['summary']}")
    print(f"Frames: {summary['artifacts']['frames']}")
    print(f"Prior: {summary['artifacts']['prior']}")
    if summary["artifacts"].get("published_prior"):
        print(f"Published prior: {summary['artifacts']['published_prior']}")
    else:
        print("Published prior: none (acceptance gate failed; diagnostic candidate only)")


if __name__ == "__main__":
    main()
