"""Internal two-stage Web job preparation and approximation-gate CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from nr_pusch.beam_validation import validate_rt_web_snapshot
from nr_pusch.config import TxSettings
from nr_pusch.rt_channel import (
    load_rt_beam_snapshot,
    prepare_rt_beam_snapshot,
    save_rt_beam_snapshot,
)
from nr_pusch.rt_config import RtBeamSettings, validate_rt_web_limits
from nr_pusch.rt_scene_assets import RtSceneAssets, resolve_scene_assets
from nr_pusch.simulation_config import BlerSettings


def _emit_stage(path: Path, stage: str) -> None:
    event = {
        "type": "stage",
        "stage": stage,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _scene_assets(settings: RtBeamSettings, root: str | None) -> RtSceneAssets | None:
    scene = settings.rt.scene
    if root is None:
        if settings.geometry is not None or scene == "custom":
            raise ValueError("parameterized/custom Web scene requires --scene-root")
        return None
    if scene == "custom" or settings.geometry is not None:
        return resolve_scene_assets(
            root,
            "scene.xml",
            source="imported" if scene == "custom" else "parameterized",
        )
    return resolve_scene_assets(root, f"{scene}.xml", source="builtin")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate an RT Web BLER job")
    parser.add_argument("--tx-config", required=True)
    parser.add_argument("--rt-config", required=True)
    parser.add_argument("--simulation-config", required=True)
    parser.add_argument("--scene-root")
    parser.add_argument("--output", required=True, help="Lower-budget RT snapshot NPZ")
    parser.add_argument("--validation-output", required=True, help="Web validation JSON report")
    parser.add_argument("--stage-jsonl", required=True, help="Append-only preparation stage events")
    args = parser.parse_args()

    validation_path = Path(args.validation_output)
    snapshot_path = Path(args.output)
    stage_path = Path(args.stage_jsonl)
    snapshot = None
    try:
        tx_settings = TxSettings.from_toml(args.tx_config)
        rt_settings = RtBeamSettings.from_toml(args.rt_config)
        simulation_settings = BlerSettings.from_toml(args.simulation_config)
        validate_rt_web_limits(rt_settings, simulation_settings)
        rt_settings.validate_transmitter(tx_settings)
        assets = _scene_assets(rt_settings, args.scene_root)

        _emit_stage(stage_path, "tracing")
        snapshot = prepare_rt_beam_snapshot(
            tx_settings, rt_settings, scene_assets=assets
        )
        save_rt_beam_snapshot(snapshot, snapshot_path)
        verified_snapshot = load_rt_beam_snapshot(
            snapshot_path,
            tx_settings,
            scene_root=assets.root if assets is not None else None,
        )

        _emit_stage(stage_path, "validating")
        asset_hashes = verified_snapshot.metadata.get("scene_asset_sha256", {})
        has_mesh = isinstance(asset_hashes, dict) and any(
            name.lower().endswith(".ply") for name in asset_hashes
        )
        higher_snapshot = None
        if has_mesh:
            from dataclasses import replace

            higher_settings = replace(
                rt_settings,
                rt=replace(
                    rt_settings.rt,
                    samples_per_src=2 * rt_settings.rt.samples_per_src,
                ),
            )
            higher_snapshot = prepare_rt_beam_snapshot(
                tx_settings, higher_settings, scene_assets=assets
            )
        report = validate_rt_web_snapshot(
            tx_settings,
            rt_settings,
            verified_snapshot,
            higher_budget_snapshot=higher_snapshot,
            device="cpu",
        )
        _write_report(validation_path, report)
        if not report["passed"]:
            print("RT Web frequency-domain acceptance gate failed", file=sys.stderr)
            return 1
        print(
            "RT Web frequency-domain acceptance gate passed; "
            f"strict_fd_td_passed={report['strict_fd_td_passed']}"
        )
        return 0
    except Exception as exc:
        report = {
            "format_version": 1,
            "policy": "web-frequency-v1",
            "passed": False,
            "strict_fd_td_passed": False,
            "checks": [{"check": "prepare_or_validation", "passed": False, "reason": str(exc)}],
            "warnings": [],
            "strict_time_report": None,
            "convergence": None,
            "reflection_oracle": {
                "applicable": False,
                "status": "not_run",
                "reason": "preparation did not reach the Web gate",
            },
            "snapshot_hash": (
                snapshot.metadata.get("array_sha256")
                if snapshot is not None else None
            ),
            "scene_bundle_sha256": (
                snapshot.metadata.get("scene_bundle_sha256")
                if snapshot is not None else None
            ),
        }
        try:
            _write_report(validation_path, report)
        except OSError:
            pass
        print(f"RT Web preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
