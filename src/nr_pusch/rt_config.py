"""Typed configuration and bounded Web limits for the four-beam Sionna RT experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import tomllib
from typing import Any, TYPE_CHECKING

from nr_pusch.config import TxSettings
from nr_pusch.rt_scene_assets import BUILTIN_SCENE_IDS

if TYPE_CHECKING:
    from .simulation_config import BlerSettings


@dataclass(frozen=True)
class RtGeometrySettings:
    ground_bounds_m: tuple[float, float, float, float] = (-500.0, 500.0, -500.0, 500.0)
    ground_height_m: float = 0.0
    ground_material: str = "concrete"
    ground_thickness_m: float = 0.1
    wall_start_xy_m: tuple[float, float] = (-100.0, 150.0)
    wall_end_xy_m: tuple[float, float] = (500.0, 150.0)
    wall_base_height_m: float = 0.0
    wall_height_m: float = 50.0
    wall_material: str = "brick"
    wall_thickness_m: float = 0.1


@dataclass(frozen=True)
class RtSceneSettings:
    scene: str = "empty"
    scene_file: str | None = None
    carrier_frequency_hz: float = 3.5e9
    synthetic_array: bool = True
    seed: int = 13
    max_depth: int = 0
    samples_per_src: int = 100_000
    max_num_paths_per_src: int = 10_000


@dataclass(frozen=True)
class RtReceiverSettings:
    name: str = "bs"
    position_m: tuple[float, float, float] = (0.0, 0.0, 25.0)
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    num_rows: int = 8
    num_cols: int = 8
    vertical_spacing_wavelengths: float = 0.5
    horizontal_spacing_wavelengths: float = 0.5
    pattern: str = "iso"
    polarization: str = "V"
    l_min: int = -6
    max_delay_spread_s: float = 3e-6


@dataclass(frozen=True)
class RtUserSettings:
    name: str
    position_m: tuple[float, float, float]
    orientation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    power_scale_db: float = 0.0


@dataclass(frozen=True)
class RtBeamformingSettings:
    azimuth_offset_deg: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    elevation_offset_deg: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    phase_bits: int = 0


@dataclass(frozen=True)
class RtNoiseSettings:
    post_combiner_ratio: float = 0.001


@dataclass(frozen=True)
class RtBeamSettings:
    rt: RtSceneSettings
    receiver: RtReceiverSettings
    users: tuple[RtUserSettings, ...]
    beams: RtBeamformingSettings
    noise: RtNoiseSettings
    geometry: RtGeometrySettings | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> "RtBeamSettings":
        with Path(path).open("rb") as stream:
            return cls.from_dict(tomllib.load(stream))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RtBeamSettings":
        raw = _table(raw, "root")
        _check_keys(raw, {"rt", "receiver", "users", "beams", "noise", "geometry"}, "root")

        rt_raw = _table(raw.get("rt", {}), "rt")
        _check_keys(
            rt_raw,
            {
                "scene", "scene_file", "carrier_frequency_hz", "synthetic_array", "seed",
                "max_depth", "samples_per_src", "max_num_paths_per_src",
            },
            "rt",
        )
        rt = RtSceneSettings(
            scene=_string(rt_raw.get("scene", "empty"), "rt.scene"),
            scene_file=(
                None if rt_raw.get("scene_file") is None
                else _string(rt_raw["scene_file"], "rt.scene_file")
            ),
            carrier_frequency_hz=_number(
                rt_raw.get("carrier_frequency_hz", 3.5e9), "rt.carrier_frequency_hz"
            ),
            synthetic_array=_boolean(rt_raw.get("synthetic_array", True), "rt.synthetic_array"),
            seed=_integer(rt_raw.get("seed", 13), "rt.seed"),
            max_depth=_integer(rt_raw.get("max_depth", 0), "rt.max_depth"),
            samples_per_src=_integer(rt_raw.get("samples_per_src", 100_000), "rt.samples_per_src"),
            max_num_paths_per_src=_integer(
                rt_raw.get("max_num_paths_per_src", 10_000), "rt.max_num_paths_per_src"
            ),
        )

        receiver_raw = _table(raw.get("receiver", {}), "receiver")
        _check_keys(
            receiver_raw,
            {
                "name", "position_m", "orientation_deg", "num_rows", "num_cols",
                "vertical_spacing_wavelengths", "horizontal_spacing_wavelengths",
                "pattern", "polarization", "l_min", "max_delay_spread_s",
            },
            "receiver",
        )
        receiver = RtReceiverSettings(
            name=_string(receiver_raw.get("name", "bs"), "receiver.name"),
            position_m=_vector3(receiver_raw.get("position_m", [0.0, 0.0, 25.0]), "receiver.position_m"),
            orientation_deg=_vector3(receiver_raw.get("orientation_deg", [0.0, 0.0, 0.0]), "receiver.orientation_deg"),
            num_rows=_integer(receiver_raw.get("num_rows", 8), "receiver.num_rows"),
            num_cols=_integer(receiver_raw.get("num_cols", 8), "receiver.num_cols"),
            vertical_spacing_wavelengths=_number(
                receiver_raw.get("vertical_spacing_wavelengths", 0.5),
                "receiver.vertical_spacing_wavelengths",
            ),
            horizontal_spacing_wavelengths=_number(
                receiver_raw.get("horizontal_spacing_wavelengths", 0.5),
                "receiver.horizontal_spacing_wavelengths",
            ),
            pattern=_string(receiver_raw.get("pattern", "iso"), "receiver.pattern"),
            polarization=_string(receiver_raw.get("polarization", "V"), "receiver.polarization"),
            l_min=_integer(receiver_raw.get("l_min", -6), "receiver.l_min"),
            max_delay_spread_s=_number(
                receiver_raw.get("max_delay_spread_s", 3e-6), "receiver.max_delay_spread_s"
            ),
        )

        users_raw = raw.get("users")
        if not isinstance(users_raw, (list, tuple)) or len(users_raw) != 4:
            raise ValueError("RT 配置必须包含恰好四个 [[users]]")
        users: list[RtUserSettings] = []
        for index, value in enumerate(users_raw):
            user_raw = _table(value, f"users[{index}]")
            _check_keys(user_raw, {"name", "position_m", "orientation_deg", "power_scale_db"}, f"users[{index}]")
            if "name" not in user_raw or "position_m" not in user_raw:
                raise ValueError(f"users[{index}] 必须设置 name 和 position_m")
            users.append(
                RtUserSettings(
                    name=_string(user_raw["name"], f"users[{index}].name"),
                    position_m=_vector3(user_raw["position_m"], f"users[{index}].position_m"),
                    orientation_deg=_vector3(
                        user_raw.get("orientation_deg", [0.0, 0.0, 0.0]),
                        f"users[{index}].orientation_deg",
                    ),
                    power_scale_db=_number(
                        user_raw.get("power_scale_db", 0.0), f"users[{index}].power_scale_db"
                    ),
                )
            )

        beams_raw = _table(raw.get("beams", {}), "beams")
        _check_keys(beams_raw, {"azimuth_offset_deg", "elevation_offset_deg", "phase_bits"}, "beams")
        beams = RtBeamformingSettings(
            azimuth_offset_deg=_vector4(
                beams_raw.get("azimuth_offset_deg", [0.0] * 4), "beams.azimuth_offset_deg"
            ),
            elevation_offset_deg=_vector4(
                beams_raw.get("elevation_offset_deg", [0.0] * 4), "beams.elevation_offset_deg"
            ),
            phase_bits=_integer(beams_raw.get("phase_bits", 0), "beams.phase_bits"),
        )

        noise_raw = _table(raw.get("noise", {}), "noise")
        _check_keys(noise_raw, {"post_combiner_ratio"}, "noise")
        noise = RtNoiseSettings(
            post_combiner_ratio=_number(
                noise_raw.get("post_combiner_ratio", 0.001), "noise.post_combiner_ratio"
            )
        )

        geometry_raw = raw.get("geometry")
        geometry = None
        if geometry_raw is not None:
            geometry_raw = _table(geometry_raw, "geometry")
            geometry_keys = {
                "ground_bounds_m", "ground_height_m", "ground_material", "ground_thickness_m",
                "wall_start_xy_m", "wall_end_xy_m", "wall_base_height_m", "wall_height_m",
                "wall_material", "wall_thickness_m",
            }
            _check_keys(geometry_raw, geometry_keys, "geometry")
            defaults = RtGeometrySettings()
            geometry = RtGeometrySettings(
                ground_bounds_m=_vector4(geometry_raw.get("ground_bounds_m", defaults.ground_bounds_m), "geometry.ground_bounds_m"),
                ground_height_m=_number(geometry_raw.get("ground_height_m", defaults.ground_height_m), "geometry.ground_height_m"),
                ground_material=_string(geometry_raw.get("ground_material", defaults.ground_material), "geometry.ground_material"),
                ground_thickness_m=_number(geometry_raw.get("ground_thickness_m", defaults.ground_thickness_m), "geometry.ground_thickness_m"),
                wall_start_xy_m=_vector2(geometry_raw.get("wall_start_xy_m", defaults.wall_start_xy_m), "geometry.wall_start_xy_m"),
                wall_end_xy_m=_vector2(geometry_raw.get("wall_end_xy_m", defaults.wall_end_xy_m), "geometry.wall_end_xy_m"),
                wall_base_height_m=_number(geometry_raw.get("wall_base_height_m", defaults.wall_base_height_m), "geometry.wall_base_height_m"),
                wall_height_m=_number(geometry_raw.get("wall_height_m", defaults.wall_height_m), "geometry.wall_height_m"),
                wall_material=_string(geometry_raw.get("wall_material", defaults.wall_material), "geometry.wall_material"),
                wall_thickness_m=_number(geometry_raw.get("wall_thickness_m", defaults.wall_thickness_m), "geometry.wall_thickness_m"),
            )
            wall_keys = {
                "wall_start_xy_m", "wall_end_xy_m", "wall_base_height_m", "wall_height_m",
                "wall_material", "wall_thickness_m",
            }
            if rt.scene == "ground" and wall_keys.intersection(geometry_raw):
                raise ValueError("ground 场景不接受 wall geometry 字段")

        settings = cls(rt=rt, receiver=receiver, users=tuple(users), beams=beams, noise=noise, geometry=geometry)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.rt.scene != "custom" and self.rt.scene not in BUILTIN_SCENE_IDS:
            raise ValueError("rt.scene 必须是内置 Sionna RT 场景或 custom")
        if self.rt.scene == "custom":
            if self.rt.scene_file != "scene.xml":
                raise ValueError("rt.scene_file 对 custom 场景必须为 scene.xml")
        elif self.rt.scene_file is not None:
            raise ValueError("rt.scene_file 仅适用于 custom 场景")
        if self.geometry is not None and self.rt.scene not in {"ground", "ground_wall"}:
            raise ValueError("geometry 仅适用于 ground 或 ground_wall 场景")
        if self.geometry is not None:
            _validate_geometry(self.geometry)
        if self.rt.carrier_frequency_hz <= 0.0:
            raise ValueError("rt.carrier_frequency_hz 必须为正有限值")
        if self.rt.seed < 0 or self.rt.max_depth < 0:
            raise ValueError("rt.seed 和 rt.max_depth 必须为非负整数")
        if self.rt.samples_per_src < 1 or self.rt.max_num_paths_per_src < 1:
            raise ValueError("RT path/sample budget 必须为正整数")
        if self.receiver.num_rows < 1 or self.receiver.num_cols < 1:
            raise ValueError("receiver.num_rows 和 num_cols 必须为正整数")
        if self.receiver.num_rows * self.receiver.num_cols < 4:
            raise ValueError("RT 接收阵元数必须至少为 4")
        if self.receiver.vertical_spacing_wavelengths <= 0.0 or self.receiver.horizontal_spacing_wavelengths <= 0.0:
            raise ValueError("receiver 阵列间距必须为正有限值")
        if self.receiver.pattern != "iso" or self.receiver.polarization != "V":
            raise ValueError("RT 基线固定使用 pattern=iso 和 polarization=V")
        if self.receiver.l_min != -6 or self.receiver.max_delay_spread_s != 3e-6:
            raise ValueError("RT DMRS tap fit 固定为 l_min=-6、max_delay_spread_s=3e-6")
        if len(self.users) != 4 or len({user.name for user in self.users}) != 4:
            raise ValueError("RT 配置必须包含四个名称唯一的用户")
        if any(not -60.0 <= user.power_scale_db <= 60.0 for user in self.users):
            raise ValueError("users.power_scale_db 必须在 [-60,60] dB")
        if not 0 <= self.beams.phase_bits <= 16:
            raise ValueError("beams.phase_bits 必须在 [0,16]")
        if not 0.0 <= self.noise.post_combiner_ratio <= 0.01:
            raise ValueError("noise.post_combiner_ratio 必须在 [0,0.01]")

    def validate_transmitter(self, tx_settings: TxSettings) -> None:
        tx_settings.validate()
        if len(tx_settings.users) != 4:
            raise ValueError("RT 四波束链路要求恰好四个 TX 用户")
        if tx_settings.pusch.num_layers != 1 or tx_settings.pusch.num_antenna_ports != 1:
            raise ValueError("RT 四波束链路要求每个 TX 用户单层且单 TX 端口")
        if any(len(user.dmrs_ports) != 1 for user in tx_settings.users):
            raise ValueError("RT 四波束链路要求每个 TX 用户一个 DMRS 端口")
        tx_names = tuple(user.name for user in tx_settings.users)
        rt_names = tuple(user.name for user in self.users)
        if tx_names != rt_names:
            raise ValueError(f"RT users 必须按 TX 用户原顺序完全匹配：TX={tx_names}, RT={rt_names}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_toml(self) -> str:
        lines: list[str] = []
        for section_name, value in (
            ("rt", self.rt), ("receiver", self.receiver), ("beams", self.beams), ("noise", self.noise)
        ):
            lines.append(f"[{section_name}]")
            for key, item in asdict(value).items():
                if item is not None:
                    lines.append(f"{key} = {_toml_value(item)}")
            lines.append("")
        for user in self.users:
            lines.append("[[users]]")
            for key, item in asdict(user).items():
                lines.append(f"{key} = {_toml_value(item)}")
            lines.append("")
        if self.geometry is not None:
            lines.append("[geometry]")
            fields = asdict(self.geometry)
            if self.rt.scene == "ground":
                fields = {key: value for key, value in fields.items() if key.startswith("ground_")}
            for key, item in fields.items():
                lines.append(f"{key} = {_toml_value(item)}")
        return "\n".join(lines).rstrip() + "\n"


def validate_rt_web_limits(
    rt_settings: RtBeamSettings,
    simulation_settings: BlerSettings | None = None,
) -> None:
    """Apply bounded local-Web limits without importing Sionna RT or a decoder."""
    rt_settings.validate()
    violations: list[str] = []
    scene = rt_settings.rt
    receiver = rt_settings.receiver
    if not 1e8 <= scene.carrier_frequency_hz <= 1e11:
        violations.append(
            f"rt.carrier_frequency_hz={scene.carrier_frequency_hz:g} outside [1e8,1e11] Hz"
        )
    if not 0 <= scene.max_depth <= 3:
        violations.append(f"rt.max_depth={scene.max_depth} outside [0,3]")
    if not 10_000 <= scene.samples_per_src <= 200_000:
        violations.append(
            f"rt.samples_per_src={scene.samples_per_src} outside [10000,200000]"
        )
    if not 1 <= scene.max_num_paths_per_src <= 10_000:
        violations.append(
            f"rt.max_num_paths_per_src={scene.max_num_paths_per_src} outside [1,10000]"
        )
    if not 1 <= receiver.num_rows <= 64:
        violations.append(f"receiver.num_rows={receiver.num_rows} outside [1,64]")
    if not 1 <= receiver.num_cols <= 64:
        violations.append(f"receiver.num_cols={receiver.num_cols} outside [1,64]")
    antenna_count = receiver.num_rows * receiver.num_cols
    if not 4 <= antenna_count <= 64:
        violations.append(f"receiver array elements={antenna_count} outside [4,64]")
    if not 0.0 < receiver.vertical_spacing_wavelengths <= 2.0:
        violations.append(
            f"receiver.vertical_spacing_wavelengths={receiver.vertical_spacing_wavelengths:g} outside (0,2]"
        )
    if not 0.0 < receiver.horizontal_spacing_wavelengths <= 2.0:
        violations.append(
            f"receiver.horizontal_spacing_wavelengths={receiver.horizontal_spacing_wavelengths:g} outside (0,2]"
        )
    if receiver.pattern != "iso":
        violations.append(f"receiver.pattern={receiver.pattern!r}; Web requires iso")
    if receiver.polarization != "V":
        violations.append(f"receiver.polarization={receiver.polarization!r}; Web requires V")
    if not scene.synthetic_array:
        violations.append("rt.synthetic_array=false; Web requires true")
    positions = [
        ("receiver.position_m", receiver.position_m),
        *((f"users[{index}].position_m", user.position_m) for index, user in enumerate(rt_settings.users)),
    ]
    for name, position in positions:
        for axis, value in zip("xyz", position):
            if abs(value) > 10_000.0:
                violations.append(f"{name}.{axis}={value:g}; absolute value exceeds 10000 m")
    for index, user in enumerate(rt_settings.users):
        distance_m = math.dist(receiver.position_m, user.position_m)
        if distance_m <= 1e-6:
            violations.append(f"users[{index}] distance from receiver={distance_m:g} m; must exceed 1e-6 m")
    if rt_settings.users[0].power_scale_db != 0.0:
        violations.append(
            f"users[0].power_scale_db={rt_settings.users[0].power_scale_db:g}; Web requires 0 dB"
        )

    if simulation_settings is not None:
        simulation_settings.validate()
        settings = simulation_settings
        if not 1 <= settings.batch_size <= 20:
            violations.append(f"bler.batch_size={settings.batch_size} outside [1,20]")
        if not 1 <= settings.max_frames_per_snr <= 2_000:
            violations.append(
                f"bler.max_frames_per_snr={settings.max_frames_per_snr} outside [1,2000]"
            )
        invalid_snrs = [snr for snr in settings.snr_db if not -30.0 <= snr <= 100.0]
        if len(settings.snr_db) > 32 or invalid_snrs:
            violations.append(
                f"bler.snr_db has {len(settings.snr_db)} points (maximum 32) or values outside [-30,100] dB: {invalid_snrs}"
            )
        if not 1 <= settings.num_decoder_iterations <= 100:
            violations.append(
                f"bler.num_decoder_iterations={settings.num_decoder_iterations} outside [1,100]"
            )
        estimators = settings.channel_estimators_for_sweep
        if len(estimators) > 2:
            violations.append(f"bler.channel_estimators has {len(estimators)} arms; maximum is 2")
        unsupported_estimators = [name for name in estimators if name not in {"perfect", "dmrs"}]
        if unsupported_estimators:
            violations.append(
                f"bler.channel_estimators contains unsupported RT arms: {unsupported_estimators}"
            )
        if len(settings.detectors) > 8:
            violations.append(f"bler.detectors has {len(settings.detectors)} arms; maximum is 8")
        if settings.channel_domain != "frequency":
            violations.append(
                f"bler.channel_domain={settings.channel_domain!r}; Web requires frequency"
            )
        if settings.l_min != receiver.l_min:
            violations.append(
                f"bler.l_min={settings.l_min}; must match receiver.l_min={receiver.l_min}"
            )
        if settings.max_delay_spread_s not in {None, receiver.max_delay_spread_s}:
            violations.append(
                "bler.max_delay_spread_s must be null or equal receiver.max_delay_spread_s"
            )
    if violations:
        raise ValueError("RT Web limits exceeded: " + "; ".join(violations))

def _validate_geometry(settings: RtGeometrySettings) -> None:
    xmin, xmax, ymin, ymax = settings.ground_bounds_m
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("geometry.ground_bounds_m 边界必须严格递增")
    if settings.wall_start_xy_m == settings.wall_end_xy_m:
        raise ValueError("geometry.wall_start_xy_m 和 wall_end_xy_m 必须不同")
    if settings.ground_material not in {"concrete", "brick"} or settings.wall_material not in {"concrete", "brick"}:
        raise ValueError("geometry material 仅支持 concrete 或 brick")
    if settings.ground_thickness_m <= 0.0 or settings.wall_thickness_m <= 0.0:
        raise ValueError("geometry thickness 必须为正数")
    if settings.ground_thickness_m > 10.0 or settings.wall_thickness_m > 10.0:
        raise ValueError("geometry thickness 最大为 10 m")
    if settings.wall_height_m <= 0.0:
        raise ValueError("geometry.wall_height_m 必须为正数")
    coordinates = (
        *settings.ground_bounds_m,
        settings.ground_height_m,
        *settings.wall_start_xy_m,
        *settings.wall_end_xy_m,
        settings.wall_base_height_m,
        settings.wall_height_m,
        settings.wall_base_height_m + settings.wall_height_m,
    )
    if any(abs(value) > 10_000.0 for value in coordinates):
        raise ValueError("geometry 坐标绝对值不得超过 10000 m")


def _table(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 TOML table 或对象")
    return value


def _check_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} 包含未知字段：{', '.join(sorted(unknown))}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限值")
    return result


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} 必须包含 {length} 个有限数值")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _vector2(value: Any, name: str) -> tuple[float, float]:
    return _vector(value, 2, name)  # type: ignore[return-value]


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    return _vector(value, 3, name)  # type: ignore[return-value]


def _vector4(value: Any, name: str) -> tuple[float, float, float, float]:
    return _vector(value, 4, name)  # type: ignore[return-value]


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = (
            f"{json.dumps(str(key), ensure_ascii=False)} = {_toml_value(item)}"
            for key, item in sorted(value.items())
        )
        return "{ " + ", ".join(pairs) + " }"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError("TOML values must be finite")
        return repr(value)
    raise TypeError(f"unsupported TOML configuration value: {type(value).__name__}")
