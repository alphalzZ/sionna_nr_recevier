"""TOML settings for SNR-versus-BLER experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class BlerSettings:
    snr_db: tuple[float, ...]
    batch_size: int
    max_frames_per_snr: int
    target_block_errors: int
    seed: int
    num_decoder_iterations: int
    channel_estimator: str
    detector: str
    detector_parameter: int | None
    detector_parameters: dict[str, int]
    detector_damping: float
    detectors: tuple[str, ...]
    device: str
    channel_domain: str = "frequency"
    detector_batch_sizes: dict[str, int] = field(default_factory=dict)
    stop_at_zero_bler: bool = False
    # Receiver-side DMRS tap window. These mirror the RX profile so a sweep can
    # reproduce the tuned capture configuration; max_delay_spread_s falls back
    # to the CDL channel value when left unset.
    l_min: int = -6
    max_delay_spread_s: float | None = None
    estimate_delay: bool = False
    channel_estimators: tuple[str, ...] | None = None

    _SUPPORTED_CHANNEL_ESTIMATORS = frozenset({"perfect", "dmrs", "dmrs-lmmse"})


    _SUPPORTED_DETECTORS = frozenset(
        {"lmmse", "lmmse-sic", "k-best", "ep", "mmse-pic", "soft-mmse-pic"}
    )

    @classmethod
    def from_toml(cls, path: str | Path) -> "BlerSettings":
        config_path = Path(path).resolve()
        with config_path.open("rb") as f:
            raw = tomllib.load(f)
        values = dict(raw["bler"])
        values["snr_db"] = tuple(float(x) for x in values["snr_db"])
        values.setdefault("detector", "lmmse")
        values.setdefault("detector_parameter", None)
        values.setdefault("detector_parameters", {})
        values.setdefault("detector_batch_sizes", {})
        values.setdefault("detector_damping", 0.25)
        values["detectors"] = tuple(values.get("detectors", (values["detector"],)))
        if "channel_estimators" in values:
            values["channel_estimators"] = tuple(values["channel_estimators"])

        values.setdefault("device", "cpu")
        values.setdefault("stop_at_zero_bler", False)
        values.setdefault("channel_domain", "frequency")
        if "dmrs_tap_power_prior_path" in values:
            raise ValueError(
                "dmrs_tap_power_prior_path 已移除；先验改为按信道配置在共享先验目录中自动查找"
            )
        settings = cls(**values)
        settings.validate()
        return settings

    @property
    def channel_estimators_for_sweep(self) -> tuple[str, ...]:
        """Return estimator arms, falling back to the legacy scalar setting."""
        if self.channel_estimators is None:
            return (self.channel_estimator,)
        return self.channel_estimators

    def validate(self) -> None:
        if not self.snr_db:
            raise ValueError("snr_db 至少需要一个扫描点")
        if any(not (float("-inf") < x < float("inf")) for x in self.snr_db):
            raise ValueError("snr_db 中不能包含 NaN 或无穷值")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size 必须为正整数")
        if not isinstance(self.max_frames_per_snr, int) or isinstance(self.max_frames_per_snr, bool) or self.max_frames_per_snr < 1:
            raise ValueError("batch_size 和 max_frames_per_snr 必须大于 0")
        if self.target_block_errors < 1 or self.num_decoder_iterations < 1:
            raise ValueError("target_block_errors 和 num_decoder_iterations 必须大于 0")
        if self.channel_estimator not in self._SUPPORTED_CHANNEL_ESTIMATORS:
            raise ValueError("channel_estimator 仅支持 perfect、dmrs 或 dmrs-lmmse")
        estimators = self.channel_estimators_for_sweep
        if (
            not isinstance(estimators, tuple)
            or not estimators
            or any(
                not isinstance(name, str) or name not in self._SUPPORTED_CHANNEL_ESTIMATORS
                for name in estimators
            )
            or len(set(estimators)) != len(estimators)
        ):
            raise ValueError(
                "channel_estimators 必须是非空且不重复的 perfect、dmrs、dmrs-lmmse 列表"
            )
        allowed = self._SUPPORTED_DETECTORS
        if self.detector not in allowed or not self.detectors or any(x not in allowed for x in self.detectors):
            raise ValueError(f"detector(s) 必须属于 {sorted(allowed)}")
        if self.detector_parameter is not None and self.detector_parameter < 1:
            raise ValueError("detector_parameter 必须大于 0")
        if any(key not in allowed or value < 1 for key, value in self.detector_parameters.items()):
            raise ValueError("detector_parameters 必须为已支持检测器配置正整数参数")
        if not isinstance(self.detector_batch_sizes, dict):
            raise ValueError("detector_batch_sizes 必须为检测器到正整数批大小的映射")
        if any(
            key not in allowed
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for key, value in self.detector_batch_sizes.items()
        ):
            raise ValueError("detector_batch_sizes 必须为已支持检测器配置正整数批大小")
        if not 0.0 < self.detector_damping <= 1.0:
            raise ValueError("detector_damping 必须位于 (0, 1]")
        if self.device not in {"cpu", "cuda", "auto"} and not self.device.startswith("cuda:"):
            raise ValueError("device 仅支持 cpu、cuda、cuda:N 或 auto")
        if self.channel_domain not in {"frequency", "time"}:
            raise ValueError("channel_domain 仅支持 frequency 或 time")
        if not isinstance(self.stop_at_zero_bler, bool):
            raise ValueError("stop_at_zero_bler 必须为布尔值")
        if isinstance(self.l_min, bool) or not isinstance(self.l_min, int):
            raise ValueError("l_min 必须是整数抽头偏移")
        if self.max_delay_spread_s is not None:
            if (
                not math.isfinite(self.max_delay_spread_s)
                or self.max_delay_spread_s <= 0
            ):
                raise ValueError("max_delay_spread_s 必须是正数")
        if not isinstance(self.estimate_delay, bool):
            raise ValueError("estimate_delay 必须是布尔值")

    def batch_size_for_detector(self, detector: str | None = None) -> int:
        """Return the configured batch size for a detector, with base-size fallback."""
        return self.detector_batch_sizes.get(detector or self.detector, self.batch_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
