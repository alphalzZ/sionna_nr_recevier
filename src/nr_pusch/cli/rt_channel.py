"""Prepare a static four-beam Sionna RT channel snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from nr_pusch.config import TxSettings
from nr_pusch.rt_channel import prepare_rt_beam_snapshot, save_rt_beam_snapshot
from nr_pusch.rt_config import RtBeamSettings
from nr_pusch.rt_scene_assets import (
    generate_parameterized_scene_assets,
    resolve_scene_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace and snapshot a four-beam Sionna RT channel")
    parser.add_argument("--tx-config", required=True, help="Four-user NR PUSCH TX TOML")
    parser.add_argument("--rt-config", required=True, help="Four-beam Sionna RT TOML")
    parser.add_argument("--scene-root", help="Explicit validated scene asset directory")
    parser.add_argument("--output", required=True, help="Output NPZ snapshot path")
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix.lower() != ".npz":
        parser.error("--output 必须使用 .npz 后缀")
    tx_settings = TxSettings.from_toml(args.tx_config)
    rt_settings = RtBeamSettings.from_toml(args.rt_config)
    scene_assets = None
    if rt_settings.geometry is not None:
        scene_root = Path(args.scene_root) if args.scene_root else output.with_name(f"{output.stem}_scene")
        scene_assets = generate_parameterized_scene_assets(
            rt_settings.rt.scene, rt_settings.geometry, scene_root
        )
    elif rt_settings.rt.scene == "custom":
        if not args.scene_root:
            parser.error("custom 场景要求 --scene-root")
        scene_assets = resolve_scene_assets(args.scene_root, "scene.xml", source="imported")
    elif args.scene_root:
        scene_assets = resolve_scene_assets(
            args.scene_root, f"{rt_settings.rt.scene}.xml", source="builtin"
        )
    snapshot = prepare_rt_beam_snapshot(
        tx_settings, rt_settings, scene_assets=scene_assets
    )
    npz_path, json_path = save_rt_beam_snapshot(snapshot, output)
    print(f"RT channel snapshot: {npz_path}")
    print(f"Manifest: {json_path}")
    print(f"Scene bundle SHA-256: {snapshot.metadata['scene_bundle_sha256']}")
    print(f"Users: {', '.join(snapshot.metadata['users'])}")
    print(f"Valid paths per user: {snapshot.metadata['valid_path_count_per_user']}")
    print(
        "Array/beam CFR shapes: "
        f"{snapshot.h_ant.shape} → {snapshot.h_beam.shape}; "
        f"RT variant: {snapshot.metadata['rt_variant']}"
    )


if __name__ == "__main__":
    main()
