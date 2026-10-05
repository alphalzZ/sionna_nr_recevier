"""Render a visual-only Sionna RT scene preview for the Web interface."""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
from typing import Literal
from uuid import uuid4

import numpy as np

from .rt_config import RtBeamSettings
from .rt_scene_assets import RtSceneAssets


_PREVIEW_LOCK = threading.Lock()
_PREVIEW_RESOLUTION = (640, 400)
_PREVIEW_SAMPLES = 16
_UE_COLORS = (
    (49 / 255, 95 / 255, 223 / 255),
    (233 / 255, 120 / 255, 50 / 255),
    (139 / 255, 85 / 255, 217 / 255),
    (59 / 255, 155 / 255, 108 / 255),
)
_BS_COLOR = (25 / 255, 37 / 255, 31 / 255)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_LARGE_SCENE_SCALE_M = 5_000.0
_PREVIEW_FAR_CLIP_M = 10_000.0
_OVERVIEW_FOV_DEG = 95.0



def _camera_pose(
    scene: object,
    positions: np.ndarray,
    view: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    bounds = scene.mi_scene.bbox()
    lower = np.asarray(bounds.min.numpy(), dtype=np.float64).reshape(3)
    upper = np.asarray(bounds.max.numpy(), dtype=np.float64).reshape(3)
    valid_geometry_bounds = (
        np.isfinite(lower).all()
        and np.isfinite(upper).all()
        and np.all(upper >= lower)
    )
    if valid_geometry_bounds:
        lower = np.minimum(lower, positions.min(axis=0))
        upper = np.maximum(upper, positions.max(axis=0))
    else:
        lower = positions.min(axis=0)
        upper = positions.max(axis=0)

    center = (lower + upper) / 2.0
    extent = np.maximum(upper - lower, 1.0)
    scene_scale = float(np.linalg.norm(extent))
    if view == "top":
        distance = max(float(max(extent[0], extent[1]) * 1.8), 10.0)
        direction = np.array([0.0, 0.0, 1.0])
    else:
        distance = max(scene_scale * 1.8, 10.0)
        direction = np.array([1.0, -1.25, 0.85], dtype=np.float64)
        direction /= np.linalg.norm(direction)
    if scene_scale > _LARGE_SCENE_SCALE_M:
        # Sionna 2.2 caps Camera rendering at 10 km; wide overviews must stay inside it.
        distance = min(
            distance,
            0.55 * scene_scale,
            max(10.0, _PREVIEW_FAR_CLIP_M - 0.55 * scene_scale),
        )
    return center + direction * distance, center, scene_scale


def render_rt_scene_preview(
    assets: RtSceneAssets,
    settings: RtBeamSettings,
    *,
    view: Literal["oblique", "top"] = "oblique",
) -> bytes:
    """Render validated scene assets with the configured BS and UE markers."""
    if view not in {"oblique", "top"}:
        raise ValueError("preview view must be oblique or top")

    positions = np.asarray(
        [settings.receiver.position_m, *(user.position_m for user in settings.users)],
        dtype=np.float64,
    )
    if positions.shape != (5, 3) or not np.isfinite(positions).all():
        raise ValueError("RT preview requires one finite BS and four finite UE positions")

    with _PREVIEW_LOCK:
        import mitsuba as mi
        from sionna.rt import Camera, Receiver, Transmitter, load_scene

        with mi.util.scoped_set_variant(mi.variant()):
            scene = load_scene(str(Path(assets.root) / assets.scene_file))
            scene.frequency = settings.rt.carrier_frequency_hz
            token = uuid4().hex
            camera_position, target, scene_scale = _camera_pose(scene, positions, view)
            display_radius = max(0.006 * scene_scale, 0.75)
            bs = Receiver(
                name=f"web_preview_{token}_bs",
                position=mi.Point3f(*positions[0].tolist()),
                color=_BS_COLOR,
                display_radius=display_radius,
            )
            transmitters = [
                Transmitter(
                    name=f"web_preview_{token}_ue{index}",
                    position=mi.Point3f(*positions[index + 1].tolist()),
                    color=_UE_COLORS[index],
                    display_radius=display_radius,
                )
                for index in range(4)
            ]
            scene.add([bs, *transmitters])
            camera = Camera(
                position=mi.Point3f(*camera_position.tolist()),
                look_at=mi.Point3f(*target.tolist()),
            )
            bitmap = scene.render(
                camera=camera,
                fov=_OVERVIEW_FOV_DEG if scene_scale > _LARGE_SCENE_SCALE_M else 48.0,
                num_samples=_PREVIEW_SAMPLES,
                resolution=_PREVIEW_RESOLUTION,
                return_bitmap=True,
                show_devices=True,
                show_orientations=False,
            )
            if bitmap is None:
                raise RuntimeError("Sionna RT did not return a scene preview image")
            with tempfile.TemporaryDirectory(prefix="nr-pusch-rt-preview-") as temporary:
                output = Path(temporary) / "scene-preview.png"
                bitmap = bitmap.convert(
                    component_format=mi.Struct.Type.UInt8,
                    srgb_gamma=True,
                )
                bitmap.write(str(output))
                image = output.read_bytes()

    if not image.startswith(_PNG_SIGNATURE):
        raise RuntimeError("Sionna RT returned an invalid PNG scene preview")
    return image
