"""Typed TOML configuration for the CDL channel stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class CdlSettings:
    model: str
    direction: str
    carrier_frequency_hz: float
    delay_spread_s: float
    max_delay_spread_s: float
    min_speed_mps: float
    max_speed_mps: float
    normalize_delays: bool
    normalize_channel: bool


@dataclass(frozen=True)
class AntennaSettings:
    rx_num_rows: int
    rx_num_cols: int
    antenna_pattern: str
    polarization: str
    polarization_type: str


@dataclass(frozen=True)
class ChannelSettings:
    channel: CdlSettings
    antennas: AntennaSettings

    @classmethod
    def from_toml(cls, path: str | Path) -> "ChannelSettings":
        with Path(path).open("rb") as f:
            raw = tomllib.load(f)
        settings = cls(
            channel=CdlSettings(**raw["channel"]),
            antennas=AntennaSettings(**raw["antennas"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        c = self.channel
        a = self.antennas
        if c.model not in {"A", "B", "C", "D", "E"}:
            raise ValueError("CDL model 必须是 38.901 定义的 A、B、C、D 或 E")
        if c.direction != "uplink":
            raise ValueError("当前 4 用户 PUSCH 信道只支持 direction='uplink'")
        if c.carrier_frequency_hz <= 0 or c.delay_spread_s <= 0:
            raise ValueError("carrier_frequency_hz 和 delay_spread_s 必须大于 0")
        if c.max_delay_spread_s <= 0 or c.max_delay_spread_s < c.delay_spread_s:
            raise ValueError("max_delay_spread_s 必须不小于 delay_spread_s")
        if c.min_speed_mps < 0 or c.max_speed_mps < c.min_speed_mps:
            raise ValueError("速度范围无效：要求 0 <= min_speed_mps <= max_speed_mps")
        if a.rx_num_rows * a.rx_num_cols != 4:
            raise ValueError("首批信道配置要求 BS 接收阵列恰好包含 4 根天线")
        if a.polarization not in {"single", "dual"}:
            raise ValueError("polarization 仅支持 single 或 dual")
        if a.polarization != "single":
            raise ValueError("首批接收阵列要求单极化，以保持 4 路接收天线")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
