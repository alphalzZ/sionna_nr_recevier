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
    tx_num_rows: int = 1
    tx_num_cols: int = 1


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
            raise ValueError("PUSCH 信道只支持 direction='uplink'")
        if c.carrier_frequency_hz <= 0 or c.delay_spread_s <= 0:
            raise ValueError("carrier_frequency_hz 和 delay_spread_s 必须大于 0")
        if c.max_delay_spread_s <= 0 or c.max_delay_spread_s < c.delay_spread_s:
            raise ValueError("max_delay_spread_s 必须不小于 delay_spread_s")
        if c.min_speed_mps < 0 or c.max_speed_mps < c.min_speed_mps:
            raise ValueError("速度范围无效：要求 0 <= min_speed_mps <= max_speed_mps")
        dimensions = (a.tx_num_rows, a.tx_num_cols, a.rx_num_rows, a.rx_num_cols)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("收发阵列行列数必须为正整数")
        if a.tx_num_rows * a.tx_num_cols not in {1, 2, 4}:
            raise ValueError("发射阵列天线数必须为 1、2 或 4")
        if a.polarization not in {"single", "dual"}:
            raise ValueError("polarization 仅支持 single 或 dual")
        if a.polarization != "single":
            raise ValueError("当前阵列使用单极化；天线数由行列数决定")

    def validate_transmitter(self, tx_settings: "TxSettings") -> None:
        """Ensure the CDL UT array models every physical PUSCH antenna port."""
        self.validate()
        configured = self.antennas.tx_num_rows * self.antennas.tx_num_cols
        if configured != tx_settings.pusch.num_antenna_ports:
            raise ValueError(
                f"CDL 发射阵列天线数 {configured} 必须等于 PUSCH "
                f"num_antenna_ports {tx_settings.pusch.num_antenna_ports}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
