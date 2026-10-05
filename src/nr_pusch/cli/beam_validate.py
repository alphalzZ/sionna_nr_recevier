"""Run ordered physical, waveform, and coded-PUSCH RT validation gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from nr_pusch.beam_simulation import simulate_rt_beam_bler
from nr_pusch.beam_validation import (
    run_rt_beam_smoke_sweeps,
    validate_rt_beam_structure,
    validate_rt_los_snapshot,
    validate_rt_multipath_snapshot,
    validate_rt_noise_model,
    validate_rt_path_convergence,
    validate_rt_time_domain,
    validate_rt_uncoded,
)
from nr_pusch.config import TxSettings
from nr_pusch.rt_channel import prepare_rt_beam_snapshot
from nr_pusch.rt_config import RtBeamSettings
from nr_pusch.rt_scene_assets import (
    generate_parameterized_scene_assets,
    resolve_scene_assets,
)
from nr_pusch.simulation_config import BlerSettings


_STAGES = (
    "rt-los",
    "beam",
    "noise",
    "uncoded",
    "pusch",
    "multipath",
    "time",
    "sweeps",
    "all",
)
_SIMULATION_STAGES = {"pusch", "sweeps", "all"}
_ALL_SEQUENCE = ("rt-los", "beam", "noise", "uncoded", "pusch", "multipath", "time", "sweeps")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the Sionna RT shared-array channel and coded PUSCH arms"
    )
    parser.add_argument("--tx-config", required=True, help="Four-user NR PUSCH TX TOML")
    parser.add_argument("--rt-config", required=True, help="Four-beam Sionna RT TOML")
    parser.add_argument("--scene-root", help="Explicit RT scene asset directory")
    parser.add_argument("--stage", choices=_STAGES, required=True, help="Validation gate to run")
    parser.add_argument(
        "--simulation-config",
        help="BLER TOML; required for pusch, sweeps, and all",
    )
    parser.add_argument("--output", required=True, help="JSON validation report path")
    parser.add_argument("--device", default="cpu", help="PyTorch PHY device; RT backend is unchanged")
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix.lower() != ".json":
        parser.error("--output 必须使用 .json 后缀")
    needs_simulation = args.stage in _SIMULATION_STAGES
    if needs_simulation != (args.simulation_config is not None):
        parser.error(
            "--simulation-config 仅在 stage=pusch、sweeps 或 all 时需要且允许"
        )

    report: dict[str, Any] = {
        "format_version": 1,
        "stage": args.stage,
        "passed": False,
        "sections": {},
    }
    try:
        tx_settings = TxSettings.from_toml(args.tx_config)
        rt_settings = RtBeamSettings.from_toml(args.rt_config)
        simulation_settings = (
            BlerSettings.from_toml(args.simulation_config)
            if args.simulation_config is not None
            else None
        )
        scene_assets = None
        if rt_settings.geometry is not None:
            scene_root = Path(args.scene_root) if args.scene_root else output.with_name(f"{output.stem}_scene")
            scene_assets = generate_parameterized_scene_assets(
                rt_settings.rt.scene, rt_settings.geometry, scene_root
            )
        elif rt_settings.rt.scene == "custom":
            if not args.scene_root:
                raise ValueError("custom 场景要求 --scene-root")
            scene_assets = resolve_scene_assets(
                args.scene_root, "scene.xml", source="imported"
            )
        elif args.scene_root:
            scene_assets = resolve_scene_assets(
                args.scene_root, f"{rt_settings.rt.scene}.xml", source="builtin"
            )
        if args.stage == "sweeps" and (
            rt_settings.geometry is not None or rt_settings.rt.scene == "custom" or args.scene_root
        ):
            raise ValueError("stage=sweeps 只支持包内场景且不能指定 --scene-root")
        tx_settings.validate()
        rt_settings.validate_transmitter(tx_settings)
        if args.stage == "all" and (
            rt_settings.rt.scene != "ground_wall" or rt_settings.geometry is not None
        ):
            raise ValueError(
                "stage=all requires the unparameterized packaged ground_wall scene"
            )
        if args.stage == "all" and simulation_settings is not None:
            if (
                simulation_settings.max_frames_per_snr != 2
                or simulation_settings.target_block_errors < 8
                or simulation_settings.stop_at_zero_bler
            ):
                raise ValueError(
                    "stage=all requires the fixed two-frame smoke budget, target_block_errors>=8, "
                    "and stop_at_zero_bler=false"
                )

        snapshot = prepare_rt_beam_snapshot(
            tx_settings, rt_settings, scene_assets=scene_assets
        )
        report["configuration"] = {
            "tx_config": str(Path(args.tx_config).resolve()),
            "rt_config": rt_settings.to_dict(),
            "simulation_config": (
                simulation_settings.to_dict() if simulation_settings is not None else None
            ),
            "device": args.device,
        }
        report["snapshot"] = {
            "config_sha256": snapshot.metadata["config_sha256"],
            "scene": snapshot.metadata["scene"],
            "solver": snapshot.metadata["solver"],
            "users": snapshot.metadata["users"],
            "scene_source": snapshot.metadata["scene_source"],
            "scene_file": snapshot.metadata["scene_file"],
            "scene_bundle_sha256": snapshot.metadata["scene_bundle_sha256"],
            "valid_path_count_per_user": snapshot.metadata["valid_path_count_per_user"],
            "rt_variant": snapshot.metadata["rt_variant"],
        }

        def run_pusch() -> dict[str, Any]:
            assert simulation_settings is not None
            rows = simulate_rt_beam_bler(
                tx_settings,
                snapshot,
                simulation_settings,
                post_combiner_ratio=rt_settings.noise.post_combiner_ratio,
                device=args.device,
            )
            statuses: dict[str, int] = {}
            for row in rows:
                statuses[row.status] = statuses.get(row.status, 0) + 1
            passed = bool(rows) and all(
                row.status in {
                    "complete",
                    "infeasible_rank",
                    "singular_noise_covariance",
                }
                for row in rows
            )
            return {
                "stage": "pusch",
                "rows": [asdict(row) for row in rows],
                "status_counts": statuses,
                "passed": passed,
            }

        def run_multipath() -> dict[str, Any]:
            if rt_settings.rt.scene != "ground_wall" or rt_settings.geometry is not None:
                raise ValueError(
                    "multipath gate requires the unparameterized packaged ground_wall scene"
                )
            reflection = validate_rt_multipath_snapshot(
                tx_settings, rt_settings, snapshot
            )
            higher_settings = replace(
                rt_settings,
                rt=replace(
                    rt_settings.rt,
                    samples_per_src=2 * rt_settings.rt.samples_per_src,
                ),
            )
            higher_snapshot = prepare_rt_beam_snapshot(
                tx_settings, higher_settings, scene_assets=scene_assets
            )
            convergence = validate_rt_path_convergence(snapshot, higher_snapshot)
            return {
                "stage": "multipath",
                "reflection_oracle": reflection,
                "path_convergence": convergence,
                "higher_budget_snapshot_sha256": higher_snapshot.metadata["config_sha256"],
                "passed": reflection["passed"] and convergence["passed"],
            }

        stage_runners = {
            "rt-los": lambda: validate_rt_los_snapshot(tx_settings, rt_settings, snapshot),
            "beam": lambda: validate_rt_beam_structure(snapshot),
            "noise": lambda: validate_rt_noise_model(snapshot),
            "uncoded": lambda: validate_rt_uncoded(snapshot),
            "pusch": run_pusch,
            "multipath": run_multipath,
            "time": lambda: validate_rt_time_domain(
                tx_settings, snapshot, device=args.device
            ),
            "sweeps": lambda: run_rt_beam_smoke_sweeps(
                tx_settings, rt_settings, _required_simulation_settings(simulation_settings),
                device=args.device,
                on_progress=lambda done, total, name: print(
                    f"Sweep {done}/{total}: {name}", flush=True
                ),
            ),
        }
        selected = _ALL_SEQUENCE if args.stage == "all" else (args.stage,)
        for stage in selected:
            try:
                section = stage_runners[stage]()
            except Exception as error:
                section = {
                    "stage": stage,
                    "passed": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            report["sections"][stage] = section
            if not section.get("passed", False):
                break
        report["passed"] = (
            len(report["sections"]) == len(selected)
            and all(section.get("passed", False) for section in report["sections"].values())
        )
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"Acceptance gate: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Report: {output}")
    if not report["passed"]:
        sys.exit(1)


def _required_simulation_settings(
    settings: BlerSettings | None,
) -> BlerSettings:
    if settings is None:
        raise ValueError("simulation settings are required for this stage")
    return settings


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


if __name__ == "__main__":
    main()
