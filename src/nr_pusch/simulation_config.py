"""TOML settings for SNR-versus-BLER experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    detectors: tuple[str, ...]
    device: str

    @classmethod
    def from_toml(cls, path: str | Path) -> "BlerSettings":
        with Path(path).open("rb") as f:
            raw = tomllib.load(f)
        values = dict(raw["bler"])
        values["snr_db"] = tuple(float(x) for x in values["snr_db"])
        values.setdefault("detector", "lmmse")
        values.setdefault("detector_parameter", None)
        values["detectors"] = tuple(values.get("detectors", (values["detector"],)))
        values.setdefault("device", "cpu")
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.snr_db:
            raise ValueError("snr_db 至少需要一个扫描点")
        if any(not (float("-inf") < x < float("inf")) for x in self.snr_db):
            raise ValueError("snr_db 中不能包含 NaN 或无穷值")
        if self.batch_size < 1 or self.max_frames_per_snr < 1:
            raise ValueError("batch_size 和 max_frames_per_snr 必须大于 0")
        if self.target_block_errors < 1 or self.num_decoder_iterations < 1:
            raise ValueError("target_block_errors 和 num_decoder_iterations 必须大于 0")
        if self.channel_estimator not in {"perfect", "dmrs"}:
            raise ValueError("channel_estimator 仅支持 perfect 或 dmrs")
        allowed = {"lmmse", "lmmse-sic", "k-best", "ep", "mmse-pic"}
        if self.detector not in allowed or not self.detectors or any(x not in allowed for x in self.detectors):
            raise ValueError(f"detector(s) 必须属于 {sorted(allowed)}")
        if self.detector_parameter is not None and self.detector_parameter < 1:
            raise ValueError("detector_parameter 必须大于 0")
        if self.device not in {"cpu", "cuda", "auto"} and not self.device.startswith("cuda:"):
            raise ValueError("device 仅支持 cpu、cuda、cuda:N 或 auto")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
