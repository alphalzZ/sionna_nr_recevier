"""Run coded NR PUSCH arms against one static four-beam RT snapshot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from nr_pusch.beam_simulation import (
    BeamBlerRow,
    make_rt_beam_manifest_metadata,
    save_rt_beam_bler,
    save_rt_beam_diagnostic_captures,
    simulate_rt_beam_bler,
)
from nr_pusch.config import TxSettings
from nr_pusch.device import use_device
from nr_pusch.rt_channel import load_rt_beam_snapshot
from nr_pusch.simulation_config import BlerSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate coded PUSCH over a fixed RT beam channel")
    parser.add_argument("--tx-config", required=True, help="Four-user NR PUSCH TX TOML")
    parser.add_argument("--channel-snapshot", required=True, help="Prepared RT channel NPZ snapshot")
    parser.add_argument("--scene-root", help="Explicit RT scene asset directory for snapshot replay")
    parser.add_argument("--simulation-config", required=True, help="TOML BLER simulation settings")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured payload/noise seed")
    parser.add_argument(
        "--post-combiner-ratio", type=float, default=None,
        help="Post-combiner/array noise variance ratio; defaults to the snapshot setting",
    )
    parser.add_argument("--output", required=True, help="Output CSV path (JSON manifest is also written)")
    parser.add_argument("--device", default=None, help="Override PHY device: cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--progress-jsonl", default=None,
        help="Append one JSON object after each completed estimator/detector/SNR point",
    )
    parser.add_argument(
        "--web-validation",
        default=None,
        help="Require a matching passed web-frequency-v1 snapshot admission report",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix.lower() != ".csv":
        parser.error("--output 必须使用 .csv 后缀")

    tx_settings = TxSettings.from_toml(args.tx_config)
    settings = BlerSettings.from_toml(args.simulation_config)
    snapshot = load_rt_beam_snapshot(
        args.channel_snapshot, tx_settings, scene_root=args.scene_root
    )

    web_validation = None
    if args.web_validation is not None:
        try:
            web_validation = json.loads(
                Path(args.web_validation).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"--web-validation 无法读取 JSON 报告: {exc}")
        if (
            not isinstance(web_validation, dict)
            or web_validation.get("format_version") != 1
            or web_validation.get("policy") != "web-frequency-v1"
            or web_validation.get("passed") is not True
        ):
            parser.error("--web-validation 必须是通过的 web-frequency-v1 报告")
        if (
            web_validation.get("snapshot_hash") != snapshot.metadata.get("array_sha256")
            or web_validation.get("scene_bundle_sha256")
            != snapshot.metadata.get("scene_bundle_sha256")
        ):
            parser.error("--web-validation snapshot 或 scene bundle hash 与 channel snapshot 不匹配")

    configured_ratio = float(snapshot.metadata["rt_config"]["noise"]["post_combiner_ratio"])
    ratio = configured_ratio if args.post_combiner_ratio is None else args.post_combiner_ratio
    effective_seed = settings.seed if args.seed is None else args.seed
    device_name = use_device(args.device or settings.device)
    progress_path = Path(args.progress_jsonl) if args.progress_jsonl else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")

    def report_point(point_rows: tuple[BeamBlerRow, ...]) -> None:
        if progress_path is not None:
            record = {"type": "point", "rows": [asdict(row) for row in point_rows]}
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    rows = simulate_rt_beam_bler(
        tx_settings,
        snapshot,
        settings,
        post_combiner_ratio=ratio,
        seed=args.seed,
        device=args.device,
        on_point=report_point if progress_path is not None else None,
    )
    capture_paths = None
    if device_name == "cpu" and settings.max_frames_per_snr <= 2:
        capture_dir = output.with_name(f"{output.stem}_captures")
        capture_paths = save_rt_beam_diagnostic_captures(
            tx_settings,
            snapshot,
            settings,
            capture_dir,
            post_combiner_ratio=ratio,
            seed=effective_seed,
            device=device_name,
        )
    metadata = make_rt_beam_manifest_metadata(
        tx_settings,
        snapshot,
        settings,
        post_combiner_ratio=ratio,
        seed=effective_seed,
        device=device_name,
    )
    if web_validation is not None:
        metadata["web_validation"] = web_validation
        metadata["scene_bundle_sha256"] = snapshot.metadata["scene_bundle_sha256"]
    if capture_paths is not None:
        metadata["diagnostic_captures"] = capture_paths
    csv_path, manifest_path = save_rt_beam_bler(rows, output, metadata=metadata)
    print(
        f"Users: {len(tx_settings.users)}; snapshot: {snapshot.metadata['scene']}; "
        f"ratio: {ratio:g}; seed: {effective_seed}; device: {device_name}"
    )
    print("Estimator       Detector           SNR [dB]  User  BLER      TB errors / TBs  Status")
    for row in rows:
        if row.user != "all":
            continue
        bler = "null" if row.bler is None else f"{row.bler:.4g}"
        print(
            f"{row.channel_estimator:13s} {row.detector:18s} {row.snr_db:8.2f}  "
            f"{row.user:4s}  {bler:8s}  {row.block_errors}/{row.transport_blocks}  {row.status}"
        )
    print(f"CSV: {csv_path}")
    print(f"Manifest: {manifest_path}")
    if progress_path is not None:
        print(f"Progress: {progress_path}")
    if capture_paths is not None:
        print(f"Diagnostic captures: {capture_paths}")


if __name__ == "__main__":
    main()
